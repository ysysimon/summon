"""Optimization ordering, independent of geometric acceptance."""
import heapq
import math
TOLERANCE_M = 1e-5
VERSION = "collision-first-v1"

def choose(choices, score):
    rows=[(r,tuple(score(r.x))) for r in choices]
    d=min(s[0] for _,s in rows)
    rows=[(r,s) for r,s in rows if s[0]<=d+TOLERANCE_M]
    rms=min(s[1] for _,s in rows)
    rows=[(r,s) for r,s in rows if s[1]<=rms+TOLERANCE_M]
    return min(rows,key=lambda row:(row[1][2:],row[1][:2]))[0]

def candidate_key(candidate):
    return tuple(candidate.get("priority_score",[candidate["loss"]]))

def scene_score(rows, weights):
    scores=[r["priority_score"] for r in rows]
    return (max(s[0] for s in scores), math.sqrt(math.fsum(w*s[1]**2 for w,s in zip(weights,scores))),
            math.fsum(w*s[2] for w,s in zip(weights,scores)),math.fsum(w*s[3] for w,s in zip(weights,scores)))

def ranked_combinations(pools,weights):
    """Admissible componentwise bounds; scalar neighbor heaps are not valid here."""
    minima=[[min(r["priority_score"][k] for r in pool) for k in range(4)] for pool in pools]
    def bound(prefix):
        rows=[pools[i][index] for i,index in enumerate(prefix)]
        rows += [dict(priority_score=s) for s in minima[len(prefix):]]
        return scene_score(rows,weights)
    queue=[(bound(()),())]
    while queue:
        score,prefix=heapq.heappop(queue)
        if len(prefix)==len(pools): yield score,prefix
        else:
            for i in range(len(pools[len(prefix)])):
                child=prefix+(i,); heapq.heappush(queue,(bound(child),child))
