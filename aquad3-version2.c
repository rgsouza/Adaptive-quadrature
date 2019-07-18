/**
*  Author: Rayanne Souza and Marcelo Paulon 
*  Last modification: 23 Oct 2018      
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <stdlib.h>

#define  TOL 1e-16

#define EXECUTE_TASK 0
#define NO_MORE_TASKS 1
#define WORKER_AVAILABLE 2
#define ADD_TASK 3

#define DEBUG 0
#define SHOW_PROGRESS 0

int n_cores, n_task;
double function(double x);
double compute_trap_area(double l, double r);
void curve_subarea(double a, double b, double area, double *result, int maySendBackToMaster);

double start_t, end_t, total_t;

MPI_Request rq;

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

void master(stack_data *stack, double *params, double total_area, double initial_l, double initial_r) {
    MPI_Status mstatus;
    int idle = 0;

    int k = 0;
    
    double progress = 0;
    double expected = initial_r - initial_l; // delta interval

    int *availableWorkers = (int *) malloc(sizeof(int) * (n_cores - 1));

    if(availableWorkers == NULL) {
        printf("Unable to create available workers list. Exiting.\n");
        exit(-1);
    }

    for(int i = 0; i < n_cores - 1; i++) {
        availableWorkers[i] = 1;
    }

    if(SHOW_PROGRESS > 0) {
        printf("\n");
    }

    int itCount = 0;

    while(1) {
        MPI_Recv(params, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &mstatus);

        if(mstatus.MPI_TAG == WORKER_AVAILABLE) {
            total_area += params[0];
            progress += params[1];

            idle++;
            if(DEBUG >= 2) {
                printf("WILL POP %d for node %d ; Total area is currently %.18f \n", k, mstatus.MPI_SOURCE, total_area);
            }

            if(SHOW_PROGRESS > 0 && ++itCount > 1000000) {
                printf("\nProgress %.2f%% (%.2f/%.2f)\n", progress/expected * 100, progress, expected);
                itCount = 0;
            }

            availableWorkers[mstatus.MPI_SOURCE - 1] = 1;

            if(!stack_is_empty(stack)){

                stack_node *node = stack_pop(stack);

                if(DEBUG >= 2) {
                    printf("POPPED %d for node %d ; Total area is currently %.18f \n", k, mstatus.MPI_SOURCE, total_area);
                }

                if (node == NULL) {
                    printf("Error - node is null");
                    exit(-1);
                }

                if(DEBUG >= 2) {
                    printf("WILL SEND %d for node %d ; Total area is currently %.18f \n", k, mstatus.MPI_SOURCE, total_area);
                }
                idle--;
                availableWorkers[mstatus.MPI_SOURCE - 1] = 0;
                MPI_Send(node, 2, MPI_DOUBLE, mstatus.MPI_SOURCE, EXECUTE_TASK, MPI_COMM_WORLD);

                if(DEBUG >= 2) {
                    printf("SENT %d for node %d ; Total area is currently %.18f \n", k, mstatus.MPI_SOURCE, total_area);
                }

            }

            if(DEBUG > 0) {
                k++;
            }
        }
        else if(mstatus.MPI_TAG == ADD_TASK) {
            if(idle){
                idle--;

                int nextAvailableWorker = -1;
                for(int i = 0; i < n_cores - 1; i++) {
                    if(availableWorkers[i] == 1) {
                        nextAvailableWorker = i + 1;
                        break;
                    }
                }

                if(nextAvailableWorker == -1) {
                    printf("Error - idle indicated there was an available worker, but it was not found on available workers list. Exiting.\n");
                    exit(-1);
                }

                MPI_Send(params, 2, MPI_DOUBLE, nextAvailableWorker, EXECUTE_TASK, MPI_COMM_WORLD);
            }
            else {
                stack_push(stack, params[0], params[1]);
            }
        }
        else {
            printf("Unknown tag. Exiting.\n");
            exit(-1);
        }

        if(DEBUG > 1) {
            printf("idle value %d \n", idle);
        }

        if(idle == (n_cores - 1) && stack_is_empty(stack) && progress/expected >= 1.0)
            break;
    }

    if(DEBUG) {
        printf("Will notify cores. \n");
    }

    // Notify all of the cores that there are no tasks left.
    for(int i = 1; i < n_cores; i++) {
        MPI_Send(&k, 2, MPI_DOUBLE, i, NO_MORE_TASKS, MPI_COMM_WORLD);
    }

    end_t = MPI_Wtime();
    printf("The area under the curve is %.16f \n", total_area);
    total_t = (double)(end_t - start_t);
    printf("Total time taken by CPU: %.16f\n", total_t);

    free(availableWorkers);
}

int main(int argc, char *argv[]){
    int p_id;
    double l, r, w, trap_area;

    double total_area = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_cores);

    if(n_cores < 2) {
        printf("A minimum of 2 cores is required for this task to work (1 master and at least one worker)");
        exit(-1);
    }

    l = atoi(argv[1]);
    r = atoi(argv[2]);
    n_task = n_cores; // We will initially start with n_cores tasks
    w = (r - l)/(n_task);

    start_t = MPI_Wtime();

    stack_data *stack = stack_create();

    for(int i = 0; i < n_task; i++) {
        double a = l + i * w;
        double b = l + (i + 1) * w;
        stack_push(stack, a, b);
    }

    double *params = (double *) malloc(sizeof(double) * 2);
    if(params == NULL) {
        printf("Unable to create params");
        exit(-1);
    }

    if(DEBUG > 0) {
        printf("Aquad3 - %d nodes ; l = %.2f ; r = %.2f - executing on node %d \n", n_cores, l, r, p_id);
    }

    if(p_id == 0) {
        master(stack, params, total_area, l, r);
    }
    else {
        if(DEBUG > 1) {
            printf("Hello %d\n", p_id);
        }

        MPI_Status status;
        double temp[2] = {0, 0};

        MPI_Send(temp, 2, MPI_DOUBLE, 0, WORKER_AVAILABLE, MPI_COMM_WORLD); // Worker initialized. Send message to master saying it's available (send 0 as previous computation)


        MPI_Recv(temp, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Wait for the task values

        while(status.MPI_TAG != NO_MORE_TASKS){
            if(status.MPI_TAG != EXECUTE_TASK) {
                printf("Unexpected tag %d, expected EXECUTE_TASK. Exiting. \n", status.MPI_TAG);
                exit(-1);
            }

            double a = temp[0];
            double b = temp[1];

            if(DEBUG >= 3) {
                printf("will execute %.2f to %.2f \n", a, b);
            }

            trap_area = compute_trap_area(a, b);

            curve_subarea(a, b, trap_area, params, 1);

            if(DEBUG) {
                printf("a = %f, b = %f \n trap area %f\n local area %f \n", a, b, trap_area, *params);
            }

            MPI_Send(params, 2, MPI_DOUBLE, 0, WORKER_AVAILABLE, MPI_COMM_WORLD);

            if(DEBUG) {
                printf("vai aguardar no receive \n");
            }

            MPI_Recv(temp, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(DEBUG) {
                printf("sai do receive  \n");
            }
        }

        if(DEBUG) {
            printf("Worker %d finished\n", p_id);
        }
    }

    MPI_Finalize();

    free(params);
    stack_destroy(stack);

    return 0;
}

double function(double x){
    double num = 10*sinh(2+10)*log(10+x*x)*cos(sqrt(pow(sqrt(1+x*x*x),3)))*atan(sqrt(2 + x*x));
    double den = pow((1 + x*x)*sqrt(2+x*x), 2)*sin(sqrt(2))*sqrt(log(2+x));

    return num/den;
}

double compute_trap_area(double l, double r) {
    return (function(l) + function(r))*(r - l)*0.5;
}

void sendBackToMaster(double a, double b) {
    double temp[2] = {a, b};
    MPI_Isend(temp, 2, MPI_DOUBLE, 0, ADD_TASK, MPI_COMM_WORLD, &rq);
}

void curve_subarea(double a, double b, double area, double *result, int maySendBackToMaster){
    double m, l_area, r_area, error;

    m = (a + b)*0.5;
    l_area = compute_trap_area(a, m);
    r_area = compute_trap_area(m, b);
    error = area - (l_area + r_area);

    if(fabs(error) <= TOL){
        result[0] = l_area + r_area;
        result[1] = b - a;
    }
    else {
        if(maySendBackToMaster) {
            sendBackToMaster(m, b);
            curve_subarea(a, m, l_area, result, 0);
        }
        else {
            curve_subarea(a, m, l_area, result, 0);
            double temp[2] = {0, 0};
            curve_subarea(m, b, r_area, temp, 0);
            result[0] += temp[0];
            result[1] += temp[1];
        }
    }
}
