"""optimizers

IMPORTANT

for many of them closure should look like this

def closure(backward=True):
    loss = ...
    if backward:
        opt.zero_grad()
        loss.backward()
    return loss


IMPORTANT

many of them SUCK (i havent tested them yet to put into BAD
"""
