import pypomp.measles.model_001b as m001b

def measles(theta):
    return pp.Pomp(
        ys=,
        theta=theta,
        covars=,
        rinit=m001b.rinit,
        rproc=m001b.rproc,
        dmeas=m001b.dmeas,
        rmeas=m001b.rmeas,
    )

