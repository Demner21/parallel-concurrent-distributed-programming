package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;
import edu.coursera.distributed.util.MPI.MPI_Request;

public class MatrixMult{
  
  public static void parallelMatrixMultiply( Matrix a, Matrix b, Matrix c, final MPI mpi ) throws MPIException{
    // Extraer el rank del actual proceso
    final int myRank = mpi.MPI_Comm_rank( mpi.MPI_COMM_WORLD );
    // Extraer el numero de ranks
    final int sizeOfRanks = mpi.MPI_Comm_size( mpi.MPI_COMM_WORLD );
    // numero de rows de la matriz que vamos a calcular
    final int numRows = c.getNRows();
    // chunks 'trozos' por cada proceso
    final int rowChunk = (numRows + sizeOfRanks - 1) / sizeOfRanks;
    int rowInicio = myRank * rowChunk;
    int rowFinal = ( myRank + 1 ) * rowChunk;
    if( rowFinal > numRows )rowFinal = numRows;
    // broadcast all of matrices a & b to all ranks
    // comuncar a todos los ranks con los valores de las matrices [tendran una copia, para poder realizar el calculo]
    mpi.MPI_Bcast( a.getValues(), 0, a.getNRows() * a.getNCols(), 0, mpi.MPI_COMM_WORLD );
    mpi.MPI_Bcast( b.getValues(), 0, b.getNRows() * b.getNCols(), 0, mpi.MPI_COMM_WORLD );
    // compute answer for rows asociados to this rank
    for( int i = rowInicio; i < rowFinal; i++ ){
      for( int j = 0; j < c.getNCols(); j++ ){
        c.set( i, j, 0.0 );
        for( int k = 0; k < b.getNRows(); k++ ){
          c.incr( i, j, a.get( i, k ) * b.get( k, j ) );
        }
      }
    }
    if( myRank == 0 ){
      MPI_Request[] requests = new MPI_Request[sizeOfRanks - 1];
      for( int i = 1; i < sizeOfRanks; i++ ){
        final int rankStartRow = i * rowChunk;
        int rankEndRow = ( i + 1 ) * rowChunk;
        if( rankEndRow > numRows )
          rankEndRow = numRows;
        final int rowOffset = rankStartRow * c.getNCols();
        final int nElements = ( rankEndRow - rankStartRow ) * c.getNCols();
        requests[i - 1] = mpi.MPI_Irecv( c.getValues(), rowOffset, nElements, i, i, mpi.MPI_COMM_WORLD );
      }
      mpi.MPI_Waitall( requests );
    }
    else{
      mpi.MPI_Send( c.getValues(), rowInicio * c.getNCols(), ( rowFinal - rowInicio ) * c.getNCols(), 0, myRank,
          mpi.MPI_COMM_WORLD );
    }
  }
}
