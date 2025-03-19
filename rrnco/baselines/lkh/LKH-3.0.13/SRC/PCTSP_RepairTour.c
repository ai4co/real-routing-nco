#include "LKH.h"
#include "Heap.h"
#include "Segment.h"

#define InDegree V

GainType PCTSP_RepairTour()
{
    int Min = INT_MAX, i = 1;
    Node *N1, *N2, *N, *CurrentRoute;
    int Forward;

//    Penalty();
    N1 = Depot;
    while ((N1 = SUCC(N1))->DepotId == 0);
    N2 = Depot;
    while ((N2 = PREDD(N2))->DepotId == 0);
    Forward = N1 != N2 ? N1->DepotId < N2->DepotId : !Reversed;
    N = Depot;
    do {
        CurrentRoute = N;
        do {
            if (N->Id <= DimensionSaved) {
                N->Loc = i++;
                N->Color = CurrentRoute->DepotId;
            }
        } while ((N = Forward ? SUCC(N) : PREDD(N))->DepotId == 0);
    } while (N != Depot);
    for (i = 1; i <= DimensionSaved; ++i) {
        N = &NodeSet[i];
        if (N->DepotId != 0)
            continue;
        if (!N->ColorAllowed[N->Color]) {
            int NewColor;
            do
                NewColor = Random() % Salesmen + 1;
            while (!N->ColorAllowed[NewColor]);
            Node *NewDepot = NewColor == 1 ? Depot :
                             &NodeSet[Dim + NewColor - 1];
            if (Forward)
                Follow(N, NewDepot)
            else
                Precede(N, NewDepot);
            N->Color = NewColor;
        }
    }

    Node *CurDepot;
    int FringeNodes;
    Constraint *Con;
    GainType Cost;
    Node **Fringe, *First = 0, *Last;

    Fringe = (Node**) malloc(DimensionSaved * sizeof(Node*));
    First = Last = Depot;
    First->Prev = First->Next = First;
    N = First;
    do
        N->InDegree = 0;
    while ((N = N->Suc) != First);
    N = CurDepot = Depot;
    do {
        CurDepot = N;
        do {
            if (N->Id <= DimensionSaved) {
                for (Con = N->FirstConstraint; Con; Con = Con->Next)
                    if (Con->t2->Color == CurDepot->DepotId)
                        Con->t2->InDegree++;
            }
        } while ((N = Forward ? SUCC(N) : PREDD(N))->DepotId == 0);
    } while (N != Depot);
    N = CurDepot = Depot;
    do {
        CurDepot = N;
        FringeNodes = 0;
        memset(Fringe, 0, DimensionSaved * sizeof(Node*));
        do {
            if (N->Id <= DimensionSaved && N->InDegree == 0 && N != CurDepot)
                Fringe[FringeNodes++] = N;
        } while ((N = Forward ? SUCC(N) : PREDD(N))->DepotId == 0);
        Node* last = N;
        Node* prev = CurDepot;
        while (FringeNodes > 0) {
            Min = prev->C[Fringe[0]->Id];
            i = 0;
            for (int j = 1; j < FringeNodes; j++) {
                if (prev->C[Fringe[j]->Id] < Min) {
                    Min = prev->C[Fringe[j]->Id];
                    i = j;
                }
            }
            assert(Fringe[i]->Color == CurDepot->DepotId);
            if (Forward) {
                Follow(Fringe[i] + DimensionSaved, prev);
                Follow(Fringe[i], Fringe[i] + DimensionSaved);
            }
            else {
                Precede(Fringe[i] + DimensionSaved, prev);
                Precede(Fringe[i], Fringe[i] + DimensionSaved);
            }
            prev = Fringe[i];
            Fringe[i] = Fringe[--FringeNodes];
            for (Con = prev->FirstConstraint; Con; Con = Con->Next) {
                if (Con->t2->Color == CurDepot->DepotId) {
                    if (--Con->t2->InDegree == 0)
                        Fringe[FringeNodes++] = Con->t2;
                    else if (Con->t2->InDegree < 0)
                        eprintf("PCTSP_RepairTour: Precedence cycle detected");
                }
            }
        }
        N = last;
    } while (N != Depot);
    free(Fringe);
    Cost = 0;
    do
        Cost += C(N, N->Suc) - N->Pi - N->Suc->Pi;
    while ((N = N->Suc) != First);
    CurrentPenalty = 0;
    return Cost / Precision;
}
