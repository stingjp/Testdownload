def powerball():
    import random
    lst = []
    while len(lst) < 5:
        x = random.randrange(1, 61)
        if x not in lst:
            lst.append(x)
    while len(lst) < 6:
        p = random.randrange(1, 37)
        if p not in lst:
            lst.append(p)

    st = ("Today's numbers are %s, %s, %s, %s, and %s. The Powerball number is %s.")
    print(st % tuple(lst))
    
    return 1


powerball()


