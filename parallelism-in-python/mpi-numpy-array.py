
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
   data = numpy.random.rand(10)
   comm.Send(data, dest=1, tag=13)
elif rank == 1:
   data = numpy.empty(10, dtype=numpy.float64)
   comm.Recv(data, source=0, tag=13)

print("rank: %d, data: %s" % (rank, data))
