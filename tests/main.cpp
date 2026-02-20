#include <gtest/gtest.h>

#include <dense_matrix.hpp>
#include <functions.hpp>

using namespace std;

constexpr float eps_float{ 1e-4 };
constexpr float eps_double{ 1e-10 };

TEST( non_singular_linear_equation_real_double, LU_decomposition_Gauss )
{
	constexpr size_t MATRIX_ROW_SIZE = 4;
	constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix< double > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< double > b( MATRIX_ROW_SIZE );
	vector< double > r( MATRIX_ROW_SIZE );
	vector< double > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< double >( 0.01, 100.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
			A.set_element( generate_random< double >( 0.01, 100.0 ), row, col );
	}

	auto A_ = A;

	A.LU_decomposition( 5 );
	A.solve_LU( x, b );

	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
}
/*

TEST( non_singular_linear_equation_real_float, QR_decomposition_Householder )
{
	constexpr size_t MATRIX_ROW_SIZE = 50;
	constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix< float > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< float > b( MATRIX_ROW_SIZE );
	vector< float > r( MATRIX_ROW_SIZE );
	vector< float > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< float >( 0.01, 100.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
			A.set_element( generate_random< float >( 0.01, 100.0 ), row, col );
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );

	A.count_residual_vector( x, b, r );
	A.iterative_refinement( A_, x, b, 0.000000000001, 1000 );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_Householder )
{
	constexpr size_t MATRIX_ROW_SIZE = 50;
	constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix< double > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< double > b( MATRIX_ROW_SIZE );
	vector< double > r( MATRIX_ROW_SIZE );
	vector< double > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< double >( 0.0001, 10000.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
			A.set_element( generate_random< double >( 0.0001, 10000.0 ), row, col );
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );

	A.count_residual_vector( x, b, r );
	A.iterative_refinement( A_, x, b, 0.000000000001, 1000 );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_Householder )
{
	constexpr size_t MATRIX_ROW_SIZE = 50;
	constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix< complex< float > > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< complex< float > > b( MATRIX_ROW_SIZE );
	vector< complex< float > > r( MATRIX_ROW_SIZE );
	vector< complex< float > > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_complex_random< float >( 0.01, 100.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
			A.set_element( generate_complex_random< float >( 0.01, 100.0 ), row, col );
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );

	A.count_residual_vector( x, b, r );
	A.iterative_refinement( A_, x, b, 0.000000000001, 1000 );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_Householder )
{
	constexpr size_t MATRIX_ROW_SIZE = 50;
	constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix< complex< double > > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< complex< double > > b( MATRIX_ROW_SIZE );
	vector< complex< double > > r( MATRIX_ROW_SIZE );
	vector< complex< double > > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_complex_random< double >( 0.0001, 10000.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
			A.set_element( generate_complex_random< double >( 0.0001, 10000.0 ), row, col );
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );

	A.count_residual_vector( x, b, r );
	A.iterative_refinement( A_, x, b, 0.000000000001, 1000 );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
}

*/