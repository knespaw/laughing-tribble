#pragma once


#include <boost/uuid.hpp>
#include <boost/uuid/random_generator.hpp>



class UuidGenerator
{
	public:

		UuidGenerator() { generator_ = boost::uuids::random_generator(); }

		boost::uuids::uuid generate() { return generator_(); }

	private:

		boost::uuids::random_generator generator_;
};