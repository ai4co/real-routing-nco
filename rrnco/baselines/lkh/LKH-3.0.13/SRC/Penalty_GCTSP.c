#include "LKH.h"
#include "Segment.h"

GainType Penalty_GCTSP()
{
    static Node *StartRoute = 0;
    Node *N, *N1, *N2, *NextN, *First, *Last, *CurrentRoute;
    GainType P = 0;
    int Forward;

    N1 = Depot;
    while ((N1 = SUCC(N1))->DepotId == 0);
    N2 = Depot;
    while ((N2 = PREDD(N2))->DepotId == 0);
    Forward = N1 != N2 ? N1->DepotId < N2->DepotId : !Reversed;

    if (!StartRoute)
        StartRoute = Depot;
    N = StartRoute;
    do {
        First = Last = 0;
        CurrentRoute = N;
        NextN = Forward ? SUCC(N) : PREDD(N);
        while ((N = NextN)->DepotId == 0) {
            if (!N->ColorAllowed[CurrentRoute->DepotId])
                P += 10000000000L;
            if (!First)
                First = N;
            Last = N;
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (!NextN->DepotId)
                P += (C(N, NextN) - N->Pi - NextN->Pi) / Precision;
            if (P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
        }
        if (First != Last)
            P += (C(First, Last) - First->Pi - Last->Pi) / Precision;
        if (P > CurrentPenalty ||
            (P == CurrentPenalty && CurrentGain <= 0)) {
            StartRoute = CurrentRoute;
            return CurrentPenalty + (CurrentGain > 0);
        }
    } while (N != StartRoute);
    return P;
}
