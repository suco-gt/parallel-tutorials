
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()

def matvec(comm, A, x):
    m = int(A.shape[0] / p)
    y_part = numpy.dot(A[rank * m:(rank+1)*m], x)
    y = numpy.zeros_like(x)
    comm.Allgather([y_part,  MPI.DOUBLE], [y, MPI.DOUBLE])
    return y

A = numpy.load("random-matrix.npy")
x = numpy.load("random-vector.npy")
y_mpi = matvec(comm, A, x)

if rank == 0:
    y = numpy.dot(A, x)
    print(y_mpi)
    print("sum(y - y_mpi) = %d" % ((y - y_mpi).sum()))
