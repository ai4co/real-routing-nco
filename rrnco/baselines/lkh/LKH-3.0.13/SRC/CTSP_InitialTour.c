#include "LKH.h"
#include "Segment.h"

/* The CTSP_InitialTour function computes an initial tour for a
 * colored TSP.
 */

GainType CTSP_InitialTour()
{
    Node *N, *NextN;
    GainType Cost;
    int Set;
    double EntryTime = GetTime();

    if (TraceLevel >= 1)
        printff("CTSP = ");
    assert(!Asymmetric);
    for (Set = 2; Set <= Salesmen; Set++)
        Follow(&NodeSet[Dim + Set - 1],
               Set == 2 ? Depot : &NodeSet[Dim + Set - 2]);
    N = Depot;
    do {
        NextN = N->Suc;
        if (N->DepotId == 0) {
            Set = N->Color != 0 ? N->Color : Random() % Salesmen + 1;
            Follow(N, Set == 1 ? Depot : &NodeSet[Dim + Set - 1]);
        }
    } while ((N = NextN) != Depot);
    Cost = 0;
    N = FirstNode;
    do
        Cost += C(N, N->Suc) - N->Pi - N->Suc->Pi;
    while ((N = N->Suc) != FirstNode);
    Cost /= Precision;
    CurrentPenalty = PLUS_INFINITY;
    CurrentPenalty = Penalty ? Penalty() : 0;
    if (TraceLevel >= 1) {
        printff(GainFormat "_" GainFormat, CurrentPenalty, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0) {
            if (ProblemType == CTSP || ProblemType == SOP ||
                ProblemType == PCTSP)
                printff(", Gap = %0.2f%%",
                        100.0 * (Cost - Optimum) / Optimum);
            else
                printff(", Gap = %0.2f%%",
                        (ProblemType == MSCTSP ? -1 : 1) *
                        100.0 * (CurrentPenalty - Optimum) / Optimum);
        }
        printff(", Time = %0.2f sec.\n", fabs(GetTime() - EntryTime));
    }
    return Cost;
}
