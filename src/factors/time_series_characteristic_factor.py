from src.factors.factor_interface import FactorInterface

# the cross sectional factors are good for single period slices
# but they will not help us with return attribution over time
# they will also not help us with risk decomposition

# we need to make a class that allows us to store a time series of
# cross sectional factors