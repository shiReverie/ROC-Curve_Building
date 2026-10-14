#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <limits>
#include <string>

struct Confusion {
    int tn, fp, fn, tp;
};

Confusion confusion_matrix(const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    double threshold) {
    Confusion c{ 0, 0, 0, 0 };
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        int pred = (y_scores[i] >= threshold) ? 1 : 0;
        if (y_true[i] == 0) {
            if (pred == 0) ++c.tn;
            else ++c.fp;
        }
        else {
            if (pred == 0) ++c.fn;
            else ++c.tp;
        }
    }
    return c;
}

struct RocPoint {
    double fpr;
    double tpr;
    double threshold;
};

std::vector<RocPoint> roc_curve(const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    double& auc) {
    const int n = static_cast<int>(y_true.size());
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return y_scores[a] > y_scores[b]; });

    int P = 0, N = 0;
    for (int y : y_true) {
        if (y == 1) ++P;
        else ++N;
    }

    std::vector<RocPoint> points;
    points.push_back({ 0.0, 0.0, std::numeric_limits<double>::infinity() });

    int tp = 0, fp = 0;
    int i = 0;
    while (i < n) {
        double score = y_scores[idx[i]];

        while (i < n && y_scores[idx[i]] == score) {
            if (y_true[idx[i]] == 1) ++tp;
            else ++fp;
            ++i;
        }

        double tpr = static_cast<double>(tp) / P;
        double fpr = static_cast<double>(fp) / N;
        points.push_back({ fpr, tpr, score });
    }

    auc = 0.0;
    for (std::size_t k = 1; k < points.size(); ++k) {
        double dx = points[k].fpr - points[k - 1].fpr;
        double avg_y = (points[k].tpr + points[k - 1].tpr) / 2.0;
        auc += dx * avg_y;
    }

    return points;
}

int main() {
    setlocale(LC_ALL, "Russian");

    std::vector<double> dog_probs = {
        0.95, 0.93, 0.91, 0.89, 0.87,
        0.85, 0.84, 0.82, 0.81, 0.80,
        0.79, 0.78, 0.77, 0.76, 0.75,
        0.74, 0.73, 0.72, 0.71, 0.70,
        0.69, 0.68, 0.67, 0.66, 0.65,
        0.64, 0.63, 0.62, 0.61, 0.60,
        0.59, 0.58, 0.57, 0.56, 0.55,
        0.45, 0.40, 0.35, 0.30, 0.20
    };

    std::vector<double> non_dog_probs = {
        0.49, 0.48, 0.47, 0.46, 0.45,
        0.44, 0.43, 0.42, 0.41, 0.40,
        0.39, 0.38, 0.37, 0.36, 0.35,
        0.34, 0.33, 0.32, 0.31, 0.30,
        0.29, 0.28, 0.27, 0.26, 0.25,
        0.24, 0.23, 0.22, 0.21, 0.20,
        0.19, 0.18, 0.17, 0.16, 0.15,
        0.14, 0.13, 0.12, 0.11, 0.10,
        0.09, 0.08, 0.07, 0.06, 0.05,
        0.04, 0.03, 0.02, 0.01, 0.005,
        0.002, 0.001,
        0.51, 0.53, 0.55, 0.58, 0.60,
        0.65, 0.70, 0.80
    };

    std::vector<int> y_true;
    std::vector<double> y_scores;
    y_true.reserve(100);
    y_scores.reserve(100);

    for (double p : dog_probs) {
        y_scores.push_back(p);
        y_true.push_back(1);
    }
    for (double p : non_dog_probs) {
        y_scores.push_back(p);
        y_true.push_back(0);
    }

    Confusion cm05 = confusion_matrix(y_true, y_scores, 0.5);

    std::cout << "Матрица ошибок при пороге 0.5:\n";
    std::cout << "TN = " << cm05.tn << ", FP = " << cm05.fp << "\n";
    std::cout << "FN = " << cm05.fn << ", TP = " << cm05.tp << "\n\n";

    double tpr_05 = static_cast<double>(cm05.tp) / (cm05.tp + cm05.fn);
    double fpr_05 = static_cast<double>(cm05.fp) / (cm05.fp + cm05.tn);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "TPR (Recall) = " << tpr_05 << "\n";
    std::cout << "FPR = " << fpr_05 << "\n\n";

    double roc_auc = 0.0;
    std::vector<RocPoint> roc_points = roc_curve(y_true, y_scores, roc_auc);

    std::cout << std::setprecision(4);
    std::cout << "ROC AUC = " << roc_auc << "\n\n";


    std::ofstream fout("roc_curve.csv");
    fout << "fpr,tpr,threshold\n";
    fout << std::setprecision(10);

    for (const RocPoint& p : roc_points) {
        fout << p.fpr << "," << p.tpr << "," << p.threshold << "\n";
    }
    fout.close();

    std::cout << "ROC-данные записаны в roc_curve.csv\n\n";

    std::cout << "Как меняются TPR и FPR при разных порогах:\n";
    std::cout << std::setw(8) << "Порог" << " | "
        << std::setw(4) << "TP" << " | "
        << std::setw(4) << "FP" << " | "
        << std::setw(6) << "TPR" << " | "
        << std::setw(6) << "FPR" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (double thresh : {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}) {
        Confusion c = confusion_matrix(y_true, y_scores, thresh);

        double tpr = static_cast<double>(c.tp) / (c.tp + c.fn);
        double fpr = static_cast<double>(c.fp) / (c.fp + c.tn);

        std::cout << std::fixed << std::setprecision(1)
            << std::setw(8) << thresh << " | "
            << std::setw(4) << c.tp << " | "
            << std::setw(4) << c.fp << " | "
            << std::setprecision(3)
            << std::setw(6) << tpr << " | "
            << std::setw(6) << fpr << "\n";
    }

    return 0;
}