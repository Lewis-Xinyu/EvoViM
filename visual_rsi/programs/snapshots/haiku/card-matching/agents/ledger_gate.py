"""First-card gate for the card-ledger memory agent (pure functions over CardLedger).

A first-card flip is a lure when the cell is face-down, its picture has already been seen
(on this cell or another), and the ledger lists no second cell with that picture. Flipping
it reveals nothing the ledger does not already hold. The gate only reads ledger state; it
never chooses an action.
"""


def known_cells(ledger):
    """Face-down cells whose picture was seen once and whose partner has not been seen."""
    return {n for _pid, cells in ledger.known_down() for n in cells}


def is_lure(ledger, choice, second_flip):
    """True when a committed choice is a first flip onto a known-unpaired face-down cell."""
    if second_flip or not choice:
        return False
    if choice in ledger.removed:
        return False
    return choice in known_cells(ledger)


def lure_message(ledger, choice):
    """One-time warning text for a lure choice. Generic: names only the ledger's own cell and picture."""
    idx = ledger.cell_pic.get(choice)
    pid = ledger.pictures[idx]["id"] if idx is not None else "unknown"
    return ("CARD LEDGER CHECK: your first card %s shows picture %s, which was already seen, and its partner "
            "has not been seen yet. Flipping it now reveals nothing new. Choose instead a cell from the "
            "\"face-down, no picture seen yet\" list, or one of the READY PAIRS if any are listed, then commit "
            "again with <|ACTION|>your_choice<|END|>." % (choice, pid))
