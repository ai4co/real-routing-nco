#include "LKH.h"
#include "Segment.h"

GainType Penalty_MSCTSP()
{
    static Node *StartRoute = 0;
    Node *N, *N1, *N2, *NextN, *CurrentRoute;
    GainType P = 0;
    int Forward, Min, d;

    N1 = Depot;
    while ((N1 = SUCC(N1))->DepotId == 0);
    N2 = Depot;
    while ((N2 = PREDD(N2))->DepotId == 0);
    Forward = N1 != N2 ? N1->DepotId < N2->DepotId : !Reversed;

    if (!StartRoute)
        StartRoute = Depot;
    N = StartRoute;
    Min = INT_MAX;
    do {
        CurrentRoute = N;
        do {
            if (N->ColorAllowed) {
                if (!N->ColorAllowed[CurrentRoute->DepotId])
                    P += 10000000L;
            } else if (N->Color != 0 && N->Color != CurrentRoute->DepotId)
                P += 10000000L;
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (Forbidden(N, NextN))
                P += 10000000L;
            if (P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
            d = -(C(N, NextN) - N->Pi - NextN->Pi) / Precision;
            if (d < Min)
                Min = d;
        } while ((N = NextN)->DepotId == 0);
    } while (N != StartRoute);
    P -= Min;
    return P;
}
