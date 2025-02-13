"""deepseek coded optimizers, i changed some of them

well well well. bad optimizers go there. they might still be good for specific tasks...

## maybe (unlikely) good ones

- third order one. not actually third order (i dont think). bad for minibatch. but maybe good for full batch???

- ray subspace. cos i changed it a lot.

## interesting (but bad) ones. those ones are simply interesting but completely impractical

- knot optimizer - just wtf is that

- rl - uses online reinforcement learning (not that it works well)

IMPORTANT

for many of them closure should look like this

def closure(backward=True):
    loss = ...
    if backward:
        opt.zero_grad()
        loss.backward()
    return loss

IMPORTANT

all of them SUCK dont use them

"""
