

count: int = 0


def printIndentHeader():
    nonlocal count
    for i in range(count):
        print('  ', end='')
    count += 1


def printIndentReturn():
    nonlocal count
    count -= 1
    for i in range(count):
        print('  ', end='')


def printIndentTailer():
    nonlocal count
    count -= 1
    for i in range(count):
        print('  ', end='')
