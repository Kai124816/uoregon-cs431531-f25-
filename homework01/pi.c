#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>


#define PI 3.1415926535


void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P2(int num_steps);


int main(int argc, char** argv)
{
    // get input values
    uint32_t num_steps = 100000;
    if(argc > 1) {
        num_steps = atoi(argv[1]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32"\n", num_steps);
    }
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);


    // set up timer
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // calculate in serial 
    start_t = ReadTSC();
    double Pi0 = calcPi_Serial(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi0);
    
    // calculate in parallel with integration
    start_t = ReadTSC();
    double Pi1 = calcPi_P1(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi1);


    // calculate in parallel with Monte Carlo
    start_t = ReadTSC();
    double Pi2 = calcPi_P2(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu32" guesses is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi2);

    
    return 0;
}


void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

double calcPi_Serial(int num_steps)
{
    printf("test\n");
    double pi = 0.0;
    double step = 1.0 / num_steps;
    double x;

    for(int i = 0; i < num_steps; i++){
        x = (i + 0.5) * step;
        pi += pow((1.0 - x * x), 0.5);
    }

    return pi * step * 4.0;
}

double calcPi_P1(int num_steps)
{
    double pi = 0.0;
    double step = 1.0 / (double)num_steps;

    #pragma omp parallel
    {
        double local_sum = 0.0;  

        #pragma omp for
        for (int i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            local_sum += pow((1.0 - x * x),0.5);
        }

        #pragma omp atomic
        pi += local_sum;
    }

    return pi * step * 4.0;
}


double calcPi_P2(int num_steps)
{
    double pi = 0.0;
    int inside_circle = 0;

    #pragma omp parallel
    {
        int local_inside = 0;

        #pragma omp for
        for(int i = 0; i < num_steps; i++){
            double rand_x = rand() / (double) RAND_MAX; 
            double rand_y = rand() / (double) RAND_MAX; 

            double dist_squared = pow(rand_x,2) + pow(rand_y,2);
            
            if(dist_squared <= 1.0){
                local_inside += 1;
            }
        }

        #pragma omp atomic
        inside_circle += local_inside;
    }

    pi = 4 * ((double)inside_circle / num_steps);
    return pi;
}
