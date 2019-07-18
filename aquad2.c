/**
*  Author: Rayanne Souza and Marcelo Paulon 
*  Last modification: 23 Oct 2018      
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <time.h>

#define  TOL 1e-16

#define EXECUTE_TASK 0
#define NO_MORE_TASKS 1
#define WORKER_AVAILABLE 2

#define DEBUG 0

int n_cores, n_task;
double function(double x);
double compute_trap_area(double l, double r);
double curve_subarea(double a, double b, double area);

typedef struct _node {
    double a;
    double b;

    struct _node *next;
} stack_node;

typedef struct _stack_data {
    int size;
    struct _node *top;
} stack_data;

stack_data * stack_create() {
    stack_data *stack = (stack_data *) malloc(sizeof(stack_data));
    if(stack == NULL) {
        printf("Unable to create stack");
        exit(-1);
    }

    stack->top = NULL;
    stack->size = 0;

    return stack;
}

void stack_push(stack_data *stack, double a, double b) {
    stack_node *node = (stack_node *) malloc(sizeof(stack_node));
    if(node == NULL) {
        printf("Unable to create node");
        exit(-1);
    }

    node->a = a;
    node->b = b;
    node->next = stack->top;
    stack->top = node;
    stack->size++;
}

stack_node * stack_pop(stack_data *stack) {
    if(stack->size == 0) {
        return NULL;
    }

    stack_node *node = stack->top;
    stack->top = node->next;
    stack->size--;
    return node;
}

int stack_is_empty(stack_data *stack) {
    if(stack->size == 0)
        return 1;
    return 0;
}

void stack_destroy(stack_data *stack) {
    stack_node *node = stack_pop(stack);
    while(node != NULL) {
        free(node);
        node = stack_pop(stack);
    }

    free(stack);
}

int main(int argc, char *argv[]){
    int p_id;
    double l, r, w, trap_area;
    double start_t, end_t, total_t;

    double total_area = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_cores);

    start_t = MPI_Wtime();

    if(n_cores < 2) {
        printf("A minimum of 2 cores is required for this task to work (1 master and at least one worker)");
        exit(-1);
    }

	l = atoi(argv[1]);
	r = atoi(argv[2]);
	n_task = atoi(argv[3]);
	w = (r - l)/(n_task);

    start_t = MPI_Wtime();

    stack_data *stack = stack_create();

    for(int i = 0; i < n_task; i++) {
        double a = l + i * w;
        double b = l + (i + 1) * w;
        stack_push(stack, a, b);
    }

    double *local_area = (double *) malloc(sizeof(double));
    if(local_area == NULL) {
        printf("Unable to create local_area");
        exit(-1);
    }

    if(DEBUG) {
        printf("Aquad2 - %d nodes ; l = %.2f ; r = %.2f - executing on node %d \n", n_cores, l, r, p_id);
    }

    if(p_id == 0) {
        MPI_Status mstatus;

        double k = 0;
        while(!stack_is_empty(stack)) {
            MPI_Recv(local_area, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WORKER_AVAILABLE,
                     MPI_COMM_WORLD, &mstatus);

            total_area += *local_area;

            if(DEBUG >= 2) {
                printf("WILL POP %.0f for node %d \n", k, mstatus.MPI_SOURCE);
            }

            stack_node *node = stack_pop(stack);

            if(node == NULL) {
                printf("Error - node is null");
                exit(-1);
            }

            MPI_Send(node, 2, MPI_DOUBLE, mstatus.MPI_SOURCE, EXECUTE_TASK, MPI_COMM_WORLD);

            k++;
        }

        // Wait for remaining cores to finish
        for(int i = 1; i < n_cores; i++) {
            MPI_Recv(local_area, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WORKER_AVAILABLE,
                     MPI_COMM_WORLD, &mstatus);

            total_area += *local_area;
        }

        if(DEBUG) {
            printf("Will notify cores. \n");
        }

        // Notify all of the cores that there are no tasks left.
        for(int i = 1; i < n_cores; i++) {
            MPI_Send(&k, 1, MPI_DOUBLE, i, NO_MORE_TASKS, MPI_COMM_WORLD);
        }

        end_t = MPI_Wtime();
        printf("The area under the curve is %.16f \n", total_area);
        total_t = (double)(end_t - start_t);
        printf("Total time taken by CPU: %.16f\n", total_t);
    }
    else {
        MPI_Status status;
        double temp[2] = {0, 0};

        MPI_Send(temp, 1, MPI_DOUBLE, 0, WORKER_AVAILABLE, MPI_COMM_WORLD); // Worker initialized. Send message to master saying it's available (send 0 as previous computation)

        MPI_Recv(temp, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Wait for the task values

        while(status.MPI_TAG != NO_MORE_TASKS){
            double a = temp[0];
            double b = temp[1];

            trap_area = compute_trap_area(a, b);

            *local_area = curve_subarea(a, b, trap_area);

            if(DEBUG) {
                printf("a = %f, b = %f \n trap area %f\n local area %f \n", a, b, trap_area, *local_area);
            }

            MPI_Send(local_area, 1, MPI_DOUBLE, 0, WORKER_AVAILABLE, MPI_COMM_WORLD);
            MPI_Recv(temp, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }

        if(DEBUG) {
            printf("Worker %d finished\n", p_id);
        }
    }

    MPI_Finalize();

    free(local_area);
    stack_destroy(stack);

    return 0;
}

double function(double x){
    double num = 10*sinh(2+10)*log(10+x*x)*cos(sqrt(pow(sqrt(1+x*x*x),3)))*atan(sqrt(2 + x*x));
    double den = pow((1 + x*x)*sqrt(2+x*x), 2)*sin(sqrt(2))*sqrt(log(2+x));

    return num/den;
}

double compute_trap_area(double l, double r){

    return (function(l) + function(r))*(r - l)*0.5;
}

double curve_subarea(double a, double b, double area){
    double m, l_area, r_area, error;

    m = (a + b)*0.5;
    l_area = compute_trap_area(a, m);
    r_area = compute_trap_area(m, b);
    error = area - (l_area + r_area);

    if(fabs(error) <= TOL){
        return l_area + r_area;
    }
    else {
        return curve_subarea(a, m, l_area) + curve_subarea(m, b, r_area);
    }
}


