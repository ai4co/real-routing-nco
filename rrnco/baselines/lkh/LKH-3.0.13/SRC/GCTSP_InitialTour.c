#include "LKH.h"

/* The GCTSP_InitialTour function computes an initial tour for a
 * general colored TSP.
 */

GainType GCTSP_InitialTour()
{
    Node *N;
    GainType Cost;
    int Set;
    double EntryTime = GetTime();

    if (TraceLevel >= 1)
        printff("GCTSP = ");
    assert(ProblemType == GCTSP || ProblemType == MSCTSP);
    assert(!Asymmetric);
    int OldTraceLevel = TraceLevel;
    TraceLevel = 0;
    InitialTourAlgorithm = GREEDY;
    GreedyTour();
    InitialTourAlgorithm = GCTSP_ALG;
    TraceLevel = OldTraceLevel;
    for (Set = 2; Set <= Salesmen; Set++)
        Follow(&NodeSet[Dim + Set - 1], &NodeSet[Dim + Set - 2]);
    N = FirstNode;
    do
        N->OldSuc = N->Suc;
    while ((N = N->Suc) != FirstNode);

    for (Set = 1; Set <= Salesmen; Set++) {
        N = FirstNode;
        do {
            if (N->Id < Dim && N->ColorAllowed[Set])
                Follow(N, &NodeSet[Dim + Set - 1]);
        } while ((N = N->OldSuc) != FirstNode);
    }
    Cost = 0;
    N = FirstNode;
    do
        Cost += C(N, N->Suc) - N->Pi - N->Suc->Pi;
    while ((N = N->Suc) != FirstNode);
    Cost /= Precision;
    CurrentPenalty = PLUS_INFINITY;
    CurrentPenalty = Penalty ? Penalty() : 0;
    if (TraceLevel >= 1) {
        printff(GainFormat "_" GainFormat,
                (ProblemType == MSCTSP ? -1 : 1) *
                CurrentPenalty,
                (ProblemType == MSCTSP ? -1 : 1) *
                Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0)
            printff(", Gap = %0.2f%%",
                    (ProblemType == MSCTSP ? -1 : 1) *
                    100.0 * (CurrentPenalty - Optimum) / Optimum);
        printff(", Time = %0.2f sec.\n", fabs(GetTime() - EntryTime));
    }
    return Cost;
}
