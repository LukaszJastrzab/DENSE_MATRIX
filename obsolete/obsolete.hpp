
// just test function
template< typename U >
std::vector< dense_matrix< U > > get_factors( const dense_matrix< U >& A )
{
	std::vector< dense_matrix< U > > factors;

	switch( A.m_dynamic_state )
	{
	case DYNAMIC_STATE::QHQ_DECOMPOSED:
		dense_matrix< U > H( A.m_rows, A.m_cols ), I( A.m_rows, A.m_cols ), Q( A.m_rows, A.m_cols ), QT( A.m_rows, A.m_cols );

		for( size_t i{ 0 }; i < A.m_rows; ++i )
		{
			I.set_element( U{ 1.0 }, i, i );
			Q.set_element( U{ 1.0 }, i, i );
			QT.set_element( U{ 1.0 }, i, i );
		}

		for( int r{ 0 }; r < static_cast< int >( A.m_rows ); ++r )
			for( int c{ std::max( 0, r - 1 ) }; c < A.m_cols; ++c )
				H.set_element( A.m_matrix[ r ][ c ], r, c );

		factors.push_back( std::move( H ) );

		const auto max_steps = A.m_rows - 2;

		for( size_t step{ 0 }; step < max_steps; ++step )
		{
			const size_t nstep{ step + 1 };
			dense_matrix< U > v( A.m_rows, 1 ), vT( 1, A.m_cols ), Q_k( A.m_rows, A.m_cols );

			v.set_element( A.m_v_firsts[ step ], nstep, 0 );
			for( size_t i{ nstep + 1 }; i < A.m_rows; ++i )
				v.set_element( A.m_matrix[ i ][ step ], i, 0 );

			vT.set_element( conjugate( A.m_v_firsts[ step ] ), 0, nstep );
			for( size_t i{ nstep + 1 }; i < A.m_cols; ++i )
				vT.set_element( conjugate( A.m_matrix[ i ][ step ] ), 0, i );

			Q_k = I - A.m_betas[ step ] * ( v * vT );

			Q = Q * Q_k;
			QT = Q_k * QT;
		}

		factors.push_back( std::move( Q ) );
		factors.push_back( std::move( QT ) );

		break;
	}

	return factors;
}