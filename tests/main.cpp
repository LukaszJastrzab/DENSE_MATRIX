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

