#include "LKH.h"

/*
 * The KSwapKick function makes a random walk K-swap kick, K>=4
 * (an extension of the double-bridge kick).
 *
 * The algorithm is inspired by the thesis
 *
 *    D. Richter,
 *    Toleranzen in Helsgauns Lin-Kernighan.Heuristik fur das TSP,
 *    Diplomarbeit, Martin-Luther-Universitat Halle-Wittenberg, 2006.
 *
 * and the paper
 *
 *    D. Applegate, W. Cook, and A. Rohe,
 *    Chained Lin-Kernighan for Large Traveling Salesman Problems.
 *    INFORMS Journal on Computing, 15 (1), pp. 82-92, 2003.
 */

#define WALK_STEPS 100
#define HUNT_COUNT (10 + Dimension / 1000)

static Node *RandomNode();
static Node *RandomWalkNode(Node * N);
static Node *LongEdgeNode();
static int compare(const void *Na, const void *Nb);

void KSwapKick(int K)
{
    Node **s, *N;
    int Count, i;

    s = (Node **) malloc(K * sizeof(Node *));
    Count = 0;
    N = FirstNode;
    do {
        N->Rank = ++Count;
        N->V = 0;
    } while ((N = N->Suc) != FirstNode);
    N = LongEdgeNode();
    if (!N)
        goto End_KSwapKick;
    FirstNode = s[0] = N;
    N->V = 1;
    for (i = 1; i < K; i++) {
        N = s[i] = RandomWalkNode(s[0]);
        if (!N)
            K = i;
        else
            N->V = 1;
    }
    if (K < 4)
        goto End_KSwapKick;
    qsort(s, K, sizeof(Node *), compare);
    for (i = 0; i < K; i++)
        s[i]->OldSuc = s[i]->Suc;
    for (i = 0; i < K; i++)
        Link(s[(i + 2) % K], s[i]->OldSuc);
  End_KSwapKick:
    free(s);
}

/*
 * The RandomNode function returns a random node N, for
 * which the edge (N, N->Suc) is neither a fixed edge nor
 * a common edge of tours to be merged, and N has not
 * previously been chosen.
 */

static Node *RandomNode()
{
    Node *N;
    int Count;

    if (SubproblemSize == 0)
        N = &NodeSet[1 + Random() % Dimension];
    else {
        N = FirstNode;
        for (Count = Random() % Dimension; Count > 0; Count--)
            N = N->Suc;
    }
    Count = 0;
    while ((FixedOrCommon(N, N->Suc) || N->V) && Count < Dimension) {
        N = N->Suc;
        Count++;
    }
    return Count < Dimension ? N : 0;
}

/*
 * The RandomWalkNode function makes a random walk on the
 * candidate edges starting at node N and returns a random
 * node R, for which the edge (R, R->Suc) is neither a
 * fixed edge nor a common edge of tours to be merged,
 * and R has not previously been chosen.
 */

static Node *RandomWalkNode(Node * N)
{
    Node *R = 0, *Last = 0;
    Candidate *NN;
    int Count, i;

    for (i = 1; i <= WALK_STEPS; i++) {
        Count = 0;
        for (NN = N->CandidateSet; NN->To; NN++)
            Count++;
        Count = Random() % Count;
        for (NN = N->CandidateSet; --Count > 0; NN++);
        if (NN->To != Last) {
            Last = N;
            N = NN->To;
            if (!N->V && !FixedOrCommon(N, N->Suc))
               R = N;
        }
    }
    return R ? R : RandomNode();
}

/*
 * The LongEdgeNode function condiders a small fraction of the nodes
 * and returns the one, R, that maximizes
 *
 *        C(R, R->Suc) - C(R, Nearest(R))
 */

static Node *LongEdgeNode()
{
    Node *N, *R = 0;
    int MaxG = INT_MIN, G, i;

    for (i = HUNT_COUNT; i > 0 && (N = RandomNode()); i--) {
        if ((G = C(N, N->Suc) - N->Cost) > MaxG) {
            MaxG = G;
            R = N;
        }
    }
    return R ? R : RandomNode();
}

static int compare(const void *Na, const void *Nb)
{
    return (*(Node **) Na)->Rank - (*(Node **) Nb)->Rank;
}
