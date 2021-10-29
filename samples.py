# Natural lengths of spring for different track markers

# marker (tape + sharpie) on 500 g mass
NATURAL_LENGTH_BASE_500g = 0.148
# marker (tape + sharpie) on 200 g mass
NATURAL_LENGTH_BASE_200g = 0.094

# spring constant (N/m)
SPRING_CONSTANT = 30.2970062

# uncertainty radius on position data
# TODO: actually calculate this
POS_UNCERTAINTY = 0.005

samples = [
  {
    "name": "m400g",
    "mass": 0.400, 
    "natural_length": NATURAL_LENGTH_BASE_200g,
  },
  {
    "name": "m500g",
    "mass": 0.500,
    "natural_length": NATURAL_LENGTH_BASE_500g
  },
  {
    "name": "m700g",
    "mass": 0.700,
    "natural_length": NATURAL_LENGTH_BASE_500g,
  },
  {
    "name": "m900g",
    "mass": 0.900,
    "natural_length": NATURAL_LENGTH_BASE_500g,
  },
  {
    "name": "m1000g",
    "mass": 1.000,
    "natural_length": NATURAL_LENGTH_BASE_500g,
  }
]