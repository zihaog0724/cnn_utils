#include <vector>

#define NMS_THRESH 0.45f
#define TOPK 3

bool cmp(const std::vector<float> &a, const std::vector<float> &b){
    return a[0] > b[0];
}

std::vector<std::vector<float>> HandDetection::nms(std::vector<std::vector<float>> &prior_result) {
    // prior_result: score, x1, y1, x2, y2
    std::sort(prior_result.begin(), prior_result.end(), cmp);
    int box_num = (int)prior_result.size();
    std::vector<std::vector<float>> ret;
    for (int i = 0; i < box_num; i++)
    {
        ret.push_back(prior_result[i]);
        if (prior_result.size() == 1 or ret.size() == TOPK)
        {
            break;
        }
        float area1 = (prior_result[i][3] - prior_result[i][1]) * (prior_result[i][4] - prior_result[i][2]);
        for (int j = i + 1; j < box_num; j++)
        {
            float xmin = std::max(prior_result[i][1], prior_result[j][1]);
            float ymin = std::max(prior_result[i][2], prior_result[j][2]);
            float xmax = std::min(prior_result[i][3], prior_result[j][3]);
            float ymax = std::min(prior_result[i][4], prior_result[j][4]);

            float intersection_w = xmax - xmin;
            float intersection_h = ymax - ymin;

            if (intersection_w <= 0 or intersection_h <= 0)
            {
                continue;
            }

            float area2 = (prior_result[j][3] - prior_result[j][1]) * (prior_result[j][4] - prior_result[j][2]);
            float intersection = intersection_h * intersection_w;
            float iou = intersection / (area1 + area2 - intersection);
            if (iou > NMS_THRESH)
            {
                prior_result.erase(prior_result.begin() + j);
                j = j - 1;
                box_num = (int)prior_result.size();
            }
        }
    }
    return ret;
}
