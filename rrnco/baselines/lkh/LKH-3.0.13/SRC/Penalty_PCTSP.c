#include "LKH.h"
#include "Segment.h"

static GainType Penalty_PCTSP_GCTSP();
static GainType Penalty_PCTSP_SOP();

GainType Penalty_PCTSP()
{
    GainType P_GCTSP = Penalty_PCTSP_GCTSP();
    if (P_GCTSP > CurrentPenalty)
        return P_GCTSP;
    return P_GCTSP + Penalty_PCTSP_SOP();
}

GainType Penalty_PCTSP_GCTSP()
{
    static Node* StartRoute = 0;
    Node* N, * N1, * N2, * CurrentRoute;
    GainType P = 0;
    int Forward;
    int i = 1;

    N1 = Depot;
    while ((N1 = SUCC(N1))->DepotId == 0);
    N2 = Depot;
    while ((N2 = PREDD(N2))->DepotId == 0);
    Forward = N1 != N2 ? N1->DepotId < N2->DepotId : !Reversed;
    if (!StartRoute)
        StartRoute = Depot;
    N = StartRoute;
    do {
        CurrentRoute = N;
        do {
            if (N->Id <= DimensionSaved) {
                N->Loc = i++;
                N->Color = CurrentRoute->DepotId;
                if (!N->ColorAllowed[N->Color])
                    P++;
            } else if (NodeSet[N->Id - DimensionSaved].DepotId != 0) {
                Node *Next = Forward ? SUCC(N) : PREDD(N);
                if (Next->DepotId == 0)
                    P++;
            }
        } while ((N = Forward ? SUCC(N) : PREDD(N))->DepotId == 0);
    } while (N != StartRoute);
    return P;
}

static GainType Penalty_PCTSP_SOP()
{
    Node *N;
    GainType P = 0, i = 1, j;
    Constraint* ConPred = 0, * ConSuc = 0;
    static Constraint* Con = 0;

    if (CurrentPenalty == 0) {
        if (Con && Con->t1->Loc > Con->t2->Loc &&
            Con->t1->Color == Con->t2->Color)
            return 1;
        for (i = Swaps - 1; i >= 0; i--) {
            for (j = 1; j <= 4; j++) {
                N = j == 1 ? SwapStack[i].t1 :
                    j == 2 ? SwapStack[i].t2 :
                    j == 3 ? SwapStack[i].t3 : SwapStack[i].t4;
                if (N->Id <= DimensionSaved) {
                    for (Con = N->FirstConstraint; Con; Con = Con->Next)
                        if (Con->t1->Loc > Con->t2->Loc &&
                            Con->t1->Color == Con->t2->Color)
                            return 1;
                }
            }
        }
    }
    for (Con = FirstConstraint; Con; ConPred = Con, Con = ConSuc) {
        ConSuc = Con->Suc;
        if (Con->t1->Loc > Con->t2->Loc && Con->t1->Color == Con->t2->Color) {
            if (Con != FirstConstraint) {
                ConPred->Suc = ConSuc;
                Con->Suc = FirstConstraint;
                FirstConstraint = Con;
                Con = ConPred;
            }
            if (++P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0))
                return CurrentPenalty + (CurrentGain > 0);
        }
    }
    return P;
}
