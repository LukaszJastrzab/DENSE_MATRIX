#include <gtest/gtest.h>

#include <dense_matrix.hpp>
#include <functions.hpp>

using namespace std;

constexpr double eps_float{ 1e-4 };
constexpr double eps_double{ 1e-10 };

using test_types = ::testing::Types< float, double, complex< float >, complex< double > >;
using test_complex_types = ::testing::Types< complex< float >, complex< double > >;

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
	dense_matrix< complex< double > > A_, IL;

	// needed vectors
	vector< DT > b, x, r;
	vector< complex< double > > L;

	double low_val{ 0.01 }, high_val{ 100.0 };

	// matrix size
	virtual size_t get_mx_size() { return 10; }
	// gets desire accuracy
	virtual RT get_acc() { return  static_cast< RT >( get_mx_size() ) * ( std::is_same_v< RT, float > ? 0.001 : 0.000001 ); }
	// matrix creation
	virtual void create_matrix() = 0;
	// decomposition for sinularity verification
	virtual void singularity_verification( dense_matrix< complex< double > >& A_IL )
	{
		EXPECT_THROW( A_IL.LU_decomposition( true, 0, get_acc() ), singularity_error );
	}


	void SetUp() override
	{
		if( std::is_same_v< real_type< T >::type, double > )
		{
			low_val = 0.001;
			high_val = 1000.0;
		}

		// tested matrix
		A.init( get_mx_size(), get_mx_size() );
		A_.init( get_mx_size(), get_mx_size() );
		IL.init( get_mx_size(), get_mx_size() );

		create_matrix();
	}

	void TearDown() override
	{
		// verification each computed eigen value 
		// if "l" is an eigen value then matrix (A - Il) is singular
		// so LU_decomposition should throw runtime error "obtained singular matrix"
		// =========================================================================
		for( size_t i{ 0 }; i < get_mx_size(); ++i )
		{
			for( size_t rc{ 0 }; rc < get_mx_size(); ++rc )
				IL.set_element( L[ i ], rc, rc );

			// if L is an eigenvalue of A (=A_) then matrix: A - IL should be singular
			// =======================================================================
			auto A_IL{ A_ - IL };
			singularity_verification( A_IL );
		}
	}
};

template< typename T >
class hermitian_eigenvalue_problem : public eigenvalues_test< T >
{
protected:
	// matrix creation
	virtual void create_matrix() override
	{
		for( size_t row{ 0 }; row < get_mx_size(); ++row )
		{
			auto val{ static_cast< T >( generate_random< RT >( low_val, high_val ) ) };
			A.set_element( val, row, row );
			A_.set_element( static_cast< complex< double > >( val ), row, row );

			for( size_t col{ row + 1 }; col < get_mx_size(); ++col )
			{
				auto val{ generate_random< T >( low_val, high_val ) };
				A.set_element( val, row, col );
				A.set_element( conjugate( val ), col, row );
				A_.set_element( static_cast< complex< double > >( val ), row, col );
				A_.set_element( static_cast< complex< double > >( conjugate( val ) ), col, row );
			}
		}
	}
};

TYPED_TEST_SUITE( hermitian_eigenvalue_problem, test_types );

TYPED_TEST( hermitian_eigenvalue_problem, QR_algorithm_Francis_OFF )
{
	// compute eigen values for matrix A
	EXPECT_NO_THROW( A.compute_eigenvalues_QR( L, 100, false ) );
}

TYPED_TEST( hermitian_eigenvalue_problem, QR_algorithm_Francis_ON )
{
	// compute eigen values for matrix A
	EXPECT_NO_THROW( A.compute_eigenvalues_QR( L, 100, true ) );
}

template< typename T >
class complex_eigenvalue_problem : public eigenvalues_test< T >
{
	// matrix creation
	virtual void create_matrix() override
	{
		for( size_t row{ 0 }; row < get_mx_size(); ++row )
			for( size_t col{ 0 }; col < get_mx_size(); ++col )
			{
				auto val{ generate_random< T >( low_val, high_val ) };
				A.set_element( val, row, col );
				A_.set_element( static_cast< complex< double > >( val ), row, col );
			}

	}
};

TYPED_TEST_SUITE( complex_eigenvalue_problem, test_complex_types );

TYPED_TEST( complex_eigenvalue_problem, QR_algorithm_Francis_OFF )
{
	// compute eigen values for matrix A
	EXPECT_NO_THROW( A.compute_eigenvalues_QR( L, 100, false ) );
}

TYPED_TEST( complex_eigenvalue_problem, QR_algorithm_Francis_ON )
{
	// compute eigen values for matrix A
	EXPECT_NO_THROW( A.compute_eigenvalues_QR( L, 100, true ) );
}

template< typename T >
class general_eigenvalue_problem : public eigenvalues_test< T >
{
protected:
	// matrix creation
	virtual void create_matrix() override
	{
		for( size_t row{ 0 }; row < get_mx_size(); ++row )
			for( size_t col{ 0 }; col < get_mx_size(); ++col )
			{
				auto val{ generate_random< T >( low_val, high_val ) };
				A.set_element( val, row, col );
				A_.set_element( static_cast< complex< double > >( val ), row, col );
			}
	}
};

TYPED_TEST_SUITE( general_eigenvalue_problem, test_types );

TYPED_TEST( general_eigenvalue_problem, QR_algorithm_Francis_ON )
{
	// compute eigen values for matrix A
	EXPECT_NO_THROW( A.compute_eigenvalues_QR( L, 100, true ) );
}
