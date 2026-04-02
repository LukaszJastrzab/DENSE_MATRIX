#include <gtest/gtest.h>

#include <dense_matrix.hpp>
#include <functions.hpp>

using namespace std;

constexpr double eps_float{ 1e-4 };
constexpr double eps_double{ 1e-10 };

using test_types = ::testing::Types< float, double, complex< float >, complex< double > >;


template< typename T >
class non_singular_linear_equation : public ::testing::Test
{
protected:
	// double type used in solving / refinement
	using DT = typename double_type< T >::type;

	// matrix of equation Ax=b
	dense_matrix< T > A;
	// a copy for test purposes
	dense_matrix< T > A_;
	// needed vectors
	vector< DT > b, x, r;

	double low_val{ 0.01 }, high_val{ 10.0 }, eps{ eps_float };

	virtual size_t get_mx_size() { return 50; }

	void SetUp() override
	{
		if( std::is_same_v< real_type< T >::type, double> )
		{
			low_val = 0.00001;
			high_val = 10000.0;
			eps = eps_double;
		}

		// matrix of equation Ax=b
		A.init( get_mx_size(), get_mx_size() );
		// right site of equation Ax=b
		b.resize( get_mx_size() );
		// residual vecotr 
		r.resize( get_mx_size() );
		// inital aproximation ( zero vector )
		x.resize( get_mx_size(), T{} );

		// randomize queation data
		for( size_t row{ 0 }; row < get_mx_size(); ++row )
		{
			b[ row ] = generate_random< DT >( low_val, high_val );

			for( size_t col{ 0 }; col < get_mx_size(); ++col )
				A.set_element( generate_random< T >( low_val, high_val ), row, col );
		}

		// make copy
		A_ = A;
	}

	void TearDown() override
	{
		// get most precise solution
		A.iterative_refinement( x, b, 0.000000000001, 1000, nullptr );

		// compute residual vector using decomposed matrix form
		A.count_residual_vector( x, b, r );
		EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );

		// compute residual vector using initial matrix form
		A_.count_residual_vector( x, b, r );
		EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );
	}
};

TYPED_TEST_SUITE( non_singular_linear_equation, test_types );

TYPED_TEST( non_singular_linear_equation, LU_decomposition_no_scaling )
{
	// decompose A=LU using Gauss elimination
	EXPECT_NO_THROW( A.LU_decomposition( false, 4 ) );
}

TYPED_TEST( non_singular_linear_equation, LU_decomposition_scaling )
{
	// decompose A=LU using Gauss elimination
	EXPECT_NO_THROW( A.LU_decomposition( true, 4 ) );
}

TYPED_TEST( non_singular_linear_equation, QR_decomposition_no_scaling )
{
	// decompose A=QR using Householder algorithm
	EXPECT_NO_THROW( A.QR_decomposition( false ) );
}

TYPED_TEST( non_singular_linear_equation, QR_decomposition_scaling )
{
	// decompose A=QR using Householder algorithm
	EXPECT_NO_THROW( A.QR_decomposition( true ) );
}


template< typename T >
class eigenvalues_test : public ::testing::Test
{
protected:
	// double type used in solving / refinement
	using DT = typename double_type< T >::type;
	// real type used in solving / refinement
	using RT = typename real_type< T >::type;

	// tested matrix
	dense_matrix< T > A;
	dense_matrix< complex< RT > > A_, Il;

	// needed vectors
	vector< DT > b, x, r;
	vector< complex< double > > l;

	double low_val{ 0.01 }, high_val{ 100.0 }, eps{ eps_float };

	virtual size_t get_mx_size() { return 4; }

	void SetUp() override
	{
		if( std::is_same_v< real_type< T >::type, double> )
		{
			low_val = 0.001;
			high_val = 1000.0;
			eps = eps_double;
		}
		// tested matrix
		A.init( get_mx_size(), get_mx_size() );
		A_.init( get_mx_size(), get_mx_size() );
		Il.init( get_mx_size(), get_mx_size() );

		// randomize matrix data
		for( size_t row{ 0 }; row < get_mx_size(); ++row )
			for( size_t col{ 0 }; col < get_mx_size(); ++col )
			{
				auto val{ generate_random< T >( low_val, high_val ) };
				A.set_element( val, row, col );
				A_.set_element( static_cast< complex< RT > >( val ), row, col );
			}

		//for( size_t row{ 0 }; row < get_mx_size(); ++row )
		//{
		//	A.set_element( static_cast< T >( generate_random< RT >( low_val, high_val ) ), row, row );

		//	for( size_t col{ row + 1 }; col < get_mx_size(); ++col )
		//	{
		//		auto val{ static_cast< T >( generate_random< T >( low_val, high_val ) ) };
		//		A.set_element( val, row, col );
		//		A.set_element( val, col, row );
		//	}
		//}
	}

	void TearDown() override
	{
		EXPECT_TRUE( true );
	}
};

using test_types2 = ::testing::Types< float, complex< float > >;

TYPED_TEST_SUITE( eigenvalues_test, test_types2 );

TYPED_TEST( eigenvalues_test, QHQ_decomposition )
{
	// decompose A=QHQ using Householder algorithm ( H is in Hessenberg form )
	EXPECT_NO_THROW( A.QHQ_decomposition() );

	EXPECT_NO_THROW( A.compute_eigenvalues_QR( l, numeric_limits< double >::min() ) );

	for( size_t i{ 0 }; i < get_mx_size(); ++i )
	{
		for( size_t rc{ 0 }; rc < get_mx_size(); ++rc )
			Il.set_element( l[ i ], rc, rc );

		auto A_l{ A_ - Il };

		EXPECT_NO_THROW( A_l.LU_decomposition( false ) );
		auto detA_l{ A_l.det() };

		int test = 7;
	}




	//auto factors = get_factors( A );
	//EXPECT_NO_THROW( factors[ 0 ].compute_eigenvalues_QR_( l, numeric_limits< double >::min() ) );





}