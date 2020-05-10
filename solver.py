from typing import Dict, List, Tuple
import data

_cardinals = ['air', 'earth', 'fire', 'water']
_metals = ['lead', 'tin', 'iron', 'copper', 'silver', 'gold']
def solve(puzzle: Dict[int, str]):
    # Gold "matches with itself" for the purposes of the moves list.
    # Find the current metal, the least metal present.
    metals = [m for m in puzzle.values() if m in _metals]
    metals.sort(key=lambda m: _metals.index(m))
    moves: List[Tuple[int, int]] = _solve_recurse(puzzle.copy(), metals[0], set())
    moves.reverse()
    return moves

def _atoms_match(a, b):
    # This could be turned into a lookup in a set of pairs.
    if a in _cardinals and (a == b or b == 'salt'):
        return True
    if b in _cardinals and (a == b or a == 'salt'):
        return True
    if a == 'salt' and b == 'salt':
        return True
    if (a in _metals and b == 'quicksilver') or (b in _metals and a == 'quicksilver'):
        return True
    if (a == 'vitae' and b == 'mors') or (a == 'mors' and b == 'vitae'):
        return True
    return False

def _solve_recurse(puzzle, current_metal, fail_memo):
    if not puzzle:
        return []
    fail_set = frozenset(puzzle.items())
    if fail_set in fail_memo:
        return None

    free_atoms = []
    for atom in puzzle:
        if puzzle[atom] not in _metals or puzzle[atom] == current_metal:
            # note that the dummy is always free
            neighbor_free = [n not in puzzle for n in data.neighbors[atom]]
            # allow wraparound
            neighbor_free.extend(neighbor_free[:2])
            for i in range(len(neighbor_free)):
                if neighbor_free[i:i+3] == [True, True, True]:
                    free_atoms.append(atom)
                    break

    for i, atom1 in enumerate(free_atoms):
        if current_metal == 'gold' and puzzle[atom1] == 'gold':
            del puzzle[atom1]
            moves = _solve_recurse(puzzle, None, fail_memo)
            if moves is not None:
                moves.append((atom1, atom1))
                return moves
            puzzle[atom1] = 'gold'

        for atom2 in free_atoms[i+1:]:
            if _atoms_match(puzzle[atom1], puzzle[atom2]):
                t1, t2 = puzzle[atom1], puzzle[atom2]
                next_metal = _metals[_metals.index(current_metal)+1] if t1 == current_metal or t2 == current_metal else current_metal
                del puzzle[atom1]
                del puzzle[atom2]
                moves = _solve_recurse(puzzle, next_metal, fail_memo)
                if moves is not None:
                    moves.append((atom1, atom2))
                    return moves
                puzzle[atom1] = t1
                puzzle[atom2] = t2

    fail_memo.add(fail_set)
    return None