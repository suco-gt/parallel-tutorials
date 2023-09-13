
import numpy
from mpi4py import MPI
from types import SimpleNamespace # allows you to add attributes to an object dynamically(ie no need to create a whole class template for it)

# buffer will be an object with an attribute data that can be accessed as buffer.data
buffer_obj = SimpleNamespace()

def broadcast(buffer, root):
    if root == 0:
        return 1
    else:
        return 0

# set up MPI parameters
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# set up data
if rank == 0:
    buffer_obj.data = numpy.random.rand(4)
else:
    buffer_obj.data = numpy.empty(4, dtype=numpy.float64)

print('Before: rank: %d, data: %s' % (rank, buffer_obj.data))
broadcast(buffer_obj, rank)
print('After: rank: %d, data: %s' % (rank, buffer_obj.data))
