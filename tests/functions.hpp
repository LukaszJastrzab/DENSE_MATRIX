#include <random>
#include <complex>

template< typename T >
T generate_random( double min_val, double max_val )
{
	static std::random_device rd;
	//static std::mt19937 gen( rd() );
	static std::mt19937 gen( 1234u );
	
	std::uniform_real_distribution< double > dis( min_val, max_val );
	std::uniform_int_distribution< int > sign_dis( 0, 1 );

	T sign = ( sign_dis( gen ) == 0 ) ? static_cast< T >( 1.0 ) : static_cast< T >( -1.0 );

	return static_cast< T >( dis( gen ) ) * sign;
}

template<>
std::complex< float > generate_random< std::complex< float > >( double min_val, double max_val )
{
	return std::complex< float >( generate_random< float >( min_val, max_val ), generate_random< float >( min_val, max_val ) );
}

template<>
std::complex< double > generate_random< std::complex< double > >( double min_val, double max_val )
{
	return std::complex< double >( generate_random< double >( min_val, max_val ), generate_random< double >( min_val, max_val ) );
}
