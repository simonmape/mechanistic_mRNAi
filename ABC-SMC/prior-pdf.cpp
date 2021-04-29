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

    // Read parameter from stdin
    double m, p, gm, gp, gb;
    std::cin >> m >> p >> gm >> gp >> gb;

    if (std::cin.fail())
    {
        std::cerr << "Error: could not read beta or gamma from stdin\n";
        return 1;
    }

    // Check if parameters are within bounds
    std::cout.precision(17);
    if ( (m_low <= m) && (m <= m_high) &&
         (p_low <= p) && (p <= p_high) &&
         (gm_low<=gm) && (gm<=gm_high) &&
         (gp_low<=gp) && (gp<=gp_high) &&
         (gb_low<=gb) && (gb<=gb_high))
    {
        std::cout << 1.0 / (m_high - m_low) / (p_high - p_low) / (gm_high - gm_low) / (gp_high - gp_low) / (gb_high - gb_low)<<
            std::endl;
    }
    else
    {
        std::cout << 0 << std::endl;
    }

    return 0;
}
