#include <Eigen/Core>
#include <vector>
#include <iostream>

using namespace std;

void FindMax(std::vector<std::pair<double, int> > &models,
                 int start, double score,
                 int *ex_steps, const Eigen::MatrixXd &Q,
                 std::vector<int>& ex_best_ind, double &ex_best_score, int ex_best_size, int ex_dim,
                 int step_size) {

    if (score > ex_best_score) {
        ex_best_score = score;
        ex_best_ind.resize(start);
        for (int i=0; i<start; i++)
            ex_best_ind[i] = models[i].second;
        ex_best_size = ex_best_ind.size();
    }

    if (start >= ex_dim)
        return;

    // Compute the effect of this model
    for (int i=start; i<ex_dim; i++) {
        int m_idx = models[i].second;
        double inc = 0;
        for (int j=0; j<start; j++)
            inc += Q(m_idx, models[j].second); // Aljosa: I think this is correct order!
        inc = 2.0*inc + Q(m_idx, m_idx);
        models[i].first = inc;
    }

    // Sort the remaining models according to their merit
    if (start < (ex_dim-1)) {
        std::vector<std::pair<double, int> > models_copy = models;
        std::sort(models_copy.begin()+start, models_copy.end(), std::greater< std::pair<double, int> >());
        models.clear();

        for (unsigned i=0; i<models_copy.size(); i++) {
            models.push_back(models_copy.at(i));
        }
    }

    // Try selecting the remaining models
    int step_no = 1;
    if (start < step_size)
        step_no = ex_steps[start];
    if (start + step_no > ex_dim)
        step_no = ex_dim - start;
    for (int i=start; i<(start+step_no); i++) {
        if (models[i].first > 0) {
            // Follow this branch recursively
            double inc = models[i].first;
            int idx = models[i].second;
            std::swap(models[start], models[i]);
            std::vector<std::pair<double, int> > models_x = models;
            FindMax(models_x, start+1, inc+score, ex_steps, Q, ex_best_ind, ex_best_score, ex_best_size, ex_dim, step_size);

            for (int j=(start+1); j<ex_dim; j++) {
                if (models[j].second == idx) {
                    std::swap(models[start], models[j]);
                    break;
                }
            }
        }
        else
            break;
    }

}


double SolveGreedyMultiBranch(const Eigen::MatrixXd &Q, Eigen::VectorXi &m) {
    assert(Q.rows()==Q.cols()); // Make sure it's a square matrix!
    const  int ex_dim = Q.rows();

    // Init indicator vector m
    m.resize(ex_dim);
    m.setZero();

    const int step_size = 6;
    int ex_steps[step_size];
    std::vector<int> ex_best_indices;
    ex_best_indices.clear();
    double ex_best_score = 0.0;
    int ex_best_size = 0;

    // Set number of potential search branches to a tracable value
    ex_steps[2] = 5;
    ex_steps[3] = 2;
    ex_steps[4] = ex_steps[5] = 1;
    if (ex_dim < 50) { // 50000 paths max
        ex_steps[0] = ex_steps[1] = 50;
    }
    else if (ex_dim < 70) { // 25000 paths max
        ex_steps[0] = 70;
        ex_steps[1] = 25;
    }
    else if (ex_dim < 100) { // 10000 paths max
        ex_steps[0] = 50;
        ex_steps[1] = 20;
    }
    else if (ex_dim < 250) { // 5000 paths max
        ex_steps[0] = 50;
        ex_steps[1] = 10;
    }
    else { // 2500 paths max
        ex_steps[0] = 50;
        ex_steps[1] = 5;
    }

    std::vector<std::pair<double, int> > models(ex_dim);
    for (int i=0; i<ex_dim; i++)
        models[i].second = i;


    FindMax(models, 0, 0.0, ex_steps, Q, ex_best_indices, ex_best_score, ex_best_size, ex_dim, step_size);

    for (int i=0; i<ex_best_indices.size(); i++) {
        m[ex_best_indices.at(i)] = 1;
        // Here, one could also read-out, or return, the corresponding hypo scores!
    }

    return ex_best_score;
}

extern "C" void SolveQPBOMultiBranch(double * data, double *data_ret, size_t mat_size)
{
    Eigen::MatrixXd Q(mat_size, mat_size);

    for(int i = 0; i < mat_size; ++i) {
        for(int j = 0; j < mat_size; ++j) {
            Q(i, j) = data[i*mat_size + j];
        }
    }

    Eigen::VectorXi m;
    SolveGreedyMultiBranch(Q, m);

    assert(m.size()==mat_size);

    for (int i=0; i<m.size(); i++) {
        data_ret[i] = m[i];
    }
}