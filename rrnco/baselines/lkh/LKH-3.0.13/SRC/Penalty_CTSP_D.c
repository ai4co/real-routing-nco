#include "LKH.h"
#include "Segment.h"

GainType Penalty_CTSP_D()
{
    Node *N;
    int *Frq, NLoop, p, i;
    GainType P[2] = {0};

    Frq = (int *) malloc((Groups + 1) * sizeof(int));

    for (p = 1; p >= 0; p--) {
        memset(Frq, 0, (Groups + 1) * sizeof(int));
        N = Depot;
        NLoop = 1;
        while (NLoop &&
               (N = p == 1 ? SUCC(N) : PREDD(N)) != Depot) {
            for (i = 1; i < N->Group - RelaxationLevel; i++) {
                P[p] += Frq[i];
                if (P[p] > CurrentPenalty) {
                    NLoop = 0;
                    break;
                }
            }
            Frq[N->Group]++;
        }
    }
    free(Frq);
    return P[0] < P[1] ? P[0] : P[1];
}
