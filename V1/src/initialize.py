import random
def initialize(lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre):
    sol = []
    for i in range(lay):
        var = random.sample(range(lb1,ub1),1);
        sol.append(var[0])
    for i in range(esc):
        var = random.sample(range(lb2,ub2),1);
        sol.append(var[0])
    for i in range(ts):
        var = random.sample(range(lb3,ub4),1);
        sol.append(var[0])
    for i in range(thre):
        var = random.sample(range(lb4,ub4),1) 
        sol.append(var[0])
    return sol;
