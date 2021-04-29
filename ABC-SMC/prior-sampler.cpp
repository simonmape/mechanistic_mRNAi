#include <string>
#include <iostream>
#include <chrono>
#include <random>


int main(int argc, char *argv[])
{
    // Process arguments
    if (argc != 11)
    {
        std::cerr << "Usage: " << argv[0]
            << " Supply low and high values for each of the model parameters.\n";
        return 1;
    }

    double m_low = std::stod(argv[1]);
    double m_high = std::stod(argv[2]);
    
    double p_low = std::stod(argv[3]);
    double p_high = std::stod(argv[4]);
    
    double gm_low = std::stod(argv[5]);
    double gm_high = std::stod(argv[6]);
    
    double gp_low = std::stod(argv[7]);
    double gp_high = std::stod(argv[8]);
    
    double gb_low = std::stod(argv[9]);
    double gb_high = std::stod(argv[10]);

    // Seed random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // Sample from uniform distribution for m
    std::uniform_real_distribution<double> m_distribution(m_low, m_high);
    double m_sampled = m_distribution(generator);
    
    // Sample from uniform distribution for p
    std::uniform_real_distribution<double> p_distribution(p_low, p_high);
    double p_sampled = p_distribution(generator);

    // Sample from uniform distribution for gm
    std::uniform_real_distribution<double> gm_distribution(gm_low, gm_high);
    double gm_sampled = gm_distribution(generator);

    // Sample from uniform distribution for gp
    std::uniform_real_distribution<double> gp_distribution(gp_low, gp_high);
    double gp_sampled = gp_distribution(generator);

    // Sample from uniform distribution for gb
    std::uniform_real_distribution<double> gb_distribution(gb_low, gb_high);
    double gb_sampled = gb_distribution(generator);

    // Print sampled parameters
    std::cout.precision(17);
    std::cout << m_sampled << " " << p_sampled << " " << gm_sampled << " " << gp_sampled << " " << gb_sampled << " " << seed << std::endl;

    return 0;
}
