#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <limits>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

static const bool DRAW_DEBUG = false;
static const bool DRAW_CONTENT_BOX = false;
static const bool DRAW_BUBBLE_BOX = false;

// 绿色聊天框整体左右镜像，但文字保持正常顺序
static const bool KEEP_BUBBLE_TEXT_READABLE = true;
static const Scalar DEFAULT_MIRRORED_BUBBLE_FILL(255, 255, 255);
static const double NICKNAME_FONT_SCALE = 1.5;

struct AvatarContentResult {
    Rect contentRect;   // patch 内真正可见头像内容区域
    Mat visibleMask;    // patch 内可见内容 mask，8UC1
};

struct AvatarInfo {
    Rect outerRect;           // 原始检测框
    Rect contentRectInPatch;  // patch 内内容框
    Mat patch;                // 原图抠出的 patch
    Mat visibleMask;          // patch 内可见内容 mask
};

struct BubbleInfo {
    Rect outerRect;      // 原始绿色聊天框矩形
    Mat patch;           // 原图抠出的 patch
    Mat visibleMask;     // patch 内整块聊天框 mask
    Scalar fillColor;    // 气泡内部主色，用于清除镜像后的反字
    Rect bodyRectInPatch;    // 去掉尾巴后的主体矩形
    Rect contentRectInPatch; // patch 内文字/前景包围框
    Mat contentPatch;        // patch 内原始方向的文字/前景 patch
    Mat contentMask;         // contentPatch 内可见文字/前景 mask
    Mat pasteMaskFull;       // patch 内精确贴字 mask
    Mat eraseMaskFull;       // patch 内放大后的清字 mask
    Mat bodyMaskFull;        // patch 内主体区域 mask
    Mat searchMaskFull;      // patch 内文本搜索区域 mask
    Mat coreMaskFull;        // patch 内强阈值文字主体 mask
    Mat looseMaskFull;       // patch 内弱阈值文字候选 mask
    bool hasContent = false;
};

struct BubbleContentResult {
    Rect bodyRect;
    Rect contentRect;
    Mat visibleMask;
    Mat pasteMaskFull;
    Mat eraseMaskFull;
    Mat bodyMaskFull;
    Mat searchMaskFull;
    Mat coreMaskFull;
    Mat looseMaskFull;
};

struct PlacedMask {
    Rect rect;
    Mat mask;
};

struct ProcessOptions {
    bool withNickname = false;
    string nickname;
};

struct NicknameReferenceStyle {
    bool valid = false;
    Scalar color = Scalar(133, 133, 133);
    int fontHeight = 24;
    int latinFontHeight = 24;
    int textXOffsetFromAvatarRight = 29;
    int textTopOffsetFromAvatarTop = -2;
    int bubbleXOffsetFromAvatarRight = 17;
    int bubbleYOffsetFromAvatarTop = 29;
    int nicknameToBubbleGap = 4;
    int blockGapY = 8;
    int referenceAvatarSize = 95;
    string latinFontPath;
    string cjkFontPath;
    int latinFontIndex = 0;
    int cjkFontIndex = 0;
    int referenceBubbleX = -1;
    int referenceNicknameX = -1;
};

struct NicknameSample {
    Rect avatarRect;
    Rect bubbleRect;
    Rect nicknameRect;
    Scalar color;
};

Scalar estimateBackgroundFromCorners(const Mat& patch);

bool looksLikeAvatar(const Mat& roi) {
    if (roi.empty()) return false;

    float ratio = static_cast<float>(roi.cols) / roi.rows;
    if (ratio < 0.75f || ratio > 1.35f) return false;

    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    Scalar m = mean(gray);
    if (m[0] < 140) return false;

    Mat edges;
    Canny(gray, edges, 50, 150);
    double edgeRatio = static_cast<double>(countNonZero(edges)) / (roi.rows * roi.cols);
    if (edgeRatio < 0.02) return false;

    return true;
}

Rect expandRect(const Rect& r, int pad, const Size& size) {
    Rect out(r.x - pad, r.y - pad, r.width + 2 * pad, r.height + 2 * pad);
    out &= Rect(0, 0, size.width, size.height);
    return out;
}

Rect insetRect(const Rect& r, int dx, int dy, const Size& size) {
    if (r.width <= 2 * dx || r.height <= 2 * dy) {
        return r & Rect(0, 0, size.width, size.height);
    }

    Rect out(r.x + dx, r.y + dy, r.width - 2 * dx, r.height - 2 * dy);
    out &= Rect(0, 0, size.width, size.height);
    return out;
}

Rect expandRectAsym(const Rect& r, int left, int top, int right, int bottom, const Size& size) {
    Rect out(r.x - left, r.y - top, r.width + left + right, r.height + top + bottom);
    out &= Rect(0, 0, size.width, size.height);
    return out;
}

Rect bboxFromMaskPeakThreshold(const Mat& mask, double colPeakRatio, double rowPeakRatio) {
    if (mask.empty() || mask.type() != CV_8UC1) return Rect();

    int maxColCount = 0;
    int maxRowCount = 0;
    vector<int> colCounts(mask.cols, 0);
    vector<int> rowCounts(mask.rows, 0);

    for (int x = 0; x < mask.cols; ++x) {
        colCounts[x] = countNonZero(mask.col(x));
        maxColCount = max(maxColCount, colCounts[x]);
    }

    for (int y = 0; y < mask.rows; ++y) {
        rowCounts[y] = countNonZero(mask.row(y));
        maxRowCount = max(maxRowCount, rowCounts[y]);
    }

    if (maxColCount == 0 || maxRowCount == 0) return Rect();

    int colThr = max(1, static_cast<int>(maxColCount * colPeakRatio));
    int rowThr = max(1, static_cast<int>(maxRowCount * rowPeakRatio));

    int x1 = -1, x2 = -1, y1 = -1, y2 = -1;

    for (int x = 0; x < mask.cols; ++x) {
        if (colCounts[x] >= colThr) {
            x1 = x;
            break;
        }
    }
    for (int x = mask.cols - 1; x >= 0; --x) {
        if (colCounts[x] >= colThr) {
            x2 = x;
            break;
        }
    }
    for (int y = 0; y < mask.rows; ++y) {
        if (rowCounts[y] >= rowThr) {
            y1 = y;
            break;
        }
    }
    for (int y = mask.rows - 1; y >= 0; --y) {
        if (rowCounts[y] >= rowThr) {
            y2 = y;
            break;
        }
    }

    if (x1 < 0 || x2 < x1 || y1 < 0 || y2 < y1) return Rect();
    return Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

Rect boundingRectOfMask(const Mat& mask) {
    if (mask.empty()) return Rect();
    vector<Point> nonZeroPts;
    findNonZero(mask, nonZeroPts);
    if (nonZeroPts.empty()) return Rect();
    return boundingRect(nonZeroPts);
}

bool hasAsciiLowercase(const string& text) {
    for (unsigned char ch : text) {
        if (ch >= 'a' && ch <= 'z') return true;
    }
    return false;
}

int leftmostMaskXInBand(const Mat& mask, double topRatio = 0.28, double bottomRatio = 0.72) {
    if (mask.empty()) return -1;

    int y1 = max(0, min(mask.rows - 1, static_cast<int>(std::floor(mask.rows * topRatio))));
    int y2 = max(y1 + 1, min(mask.rows, static_cast<int>(std::ceil(mask.rows * bottomRatio))));
    int bestX = mask.cols;

    for (int y = y1; y < y2; ++y) {
        const uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x]) {
                bestX = min(bestX, x);
                break;
            }
        }
    }

    if (bestX < mask.cols) return bestX;
    Rect bounds = boundingRectOfMask(mask);
    return bounds.area() > 0 ? bounds.x : -1;
}

Mat extractWhiteBubbleMask(const Mat& img, const Rect& bubbleRect) {
    if (img.empty() || bubbleRect.area() <= 0) return Mat();

    Rect roiRect = expandRect(bubbleRect, 3, img.size());
    Mat roi = img(roiRect).clone();

    Mat hsv;
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    Mat whiteMask;
    inRange(hsv, Scalar(0, 0, 215), Scalar(180, 45, 255), whiteMask);

    Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(whiteMask, whiteMask, MORPH_CLOSE, kernel3);

    Rect localBubble(bubbleRect.x - roiRect.x, bubbleRect.y - roiRect.y, bubbleRect.width, bubbleRect.height);
    localBubble &= Rect(0, 0, roi.cols, roi.rows);
    if (localBubble.area() <= 0) return whiteMask;

    vector<vector<Point>> contours;
    findContours(whiteMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double bestScore = -1.0;
    int bestIdx = -1;
    Point localCenter(localBubble.x + localBubble.width / 2, localBubble.y + localBubble.height / 2);
    for (size_t i = 0; i < contours.size(); ++i) {
        Rect r = boundingRect(contours[i]);
        double overlap = static_cast<double>((r & localBubble).area());
        bool containsCenter = pointPolygonTest(contours[i], localCenter, false) >= 0;
        double score = overlap + (containsCenter ? localBubble.area() : 0.0);
        if (score > bestScore) {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        }
    }

    if (bestIdx < 0) return whiteMask;
    Mat chosen = Mat::zeros(whiteMask.size(), CV_8UC1);
    drawContours(chosen, contours, bestIdx, Scalar(255), FILLED, LINE_8);
    return chosen;
}

vector<Rect> findRightGreenBubbles(const Mat& img) {
    vector<Rect> bubbles;

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    Scalar lower(35, 60, 120);
    Scalar upper(70, 255, 255);

    Mat mask;
    inRange(hsv, lower, upper, mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& c : contours) {
        Rect r = boundingRect(c);
        if (r.area() < 2500) continue;
        if (r.width < 50 || r.height < 35) continue;
        if (r.x + r.width < img.cols * 0.8) continue;
        bubbles.push_back(r);
    }

    sort(bubbles.begin(), bubbles.end(), [](const Rect& a, const Rect& b) {
        if (abs(a.y - b.y) > 10) return a.y < b.y;
        return a.x < b.x;
    });

    return bubbles;
}

int medianInt(vector<int> values, int fallback) {
    if (values.empty()) return fallback;
    sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int medianRectX(const vector<Rect>& rects, int fallback) {
    vector<int> xs;
    for (const auto& r : rects) {
        if (r.area() > 0) xs.push_back(r.x);
    }
    return medianInt(xs, fallback);
}

string pickNicknameLatinFontPath() {
    const vector<string> candidates = {
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) return path;
    }
    return "";
}

string pickNicknameCjkFontPath() {
    const vector<string> candidates = {
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) return path;
    }
    return "";
}

int fontFaceIndexForNicknamePath(const string& fontPath) {
    if (fontPath.find("PingFang.ttc") != string::npos) {
        return 11; // PingFang SC Semibold
    }
    return 0;
}

vector<Rect> findLeftWhiteBubbles(const Mat& img) {
    vector<Rect> bubbles;
    if (img.empty()) return bubbles;

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    Mat whiteMask;
    inRange(hsv, Scalar(0, 0, 215), Scalar(180, 42, 255), whiteMask);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(whiteMask, whiteMask, MORPH_OPEN, kernel3);
    morphologyEx(whiteMask, whiteMask, MORPH_CLOSE, kernel5);

    vector<vector<Point>> contours;
    findContours(whiteMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        if (r.area() < 1100) continue;
        if (r.width < 42 || r.height < 24) continue;
        if (r.width > img.cols * 0.75) continue;
        if (r.x > img.cols * 0.58) continue;
        if (r.y < img.rows * 0.06 || r.y > img.rows * 0.90) continue;
        if (r.y < img.rows * 0.18 && r.width > img.cols * 0.45) continue;

        float ratio = static_cast<float>(r.width) / max(1, r.height);
        if (ratio < 0.8f || ratio > 9.5f) continue;
        bubbles.push_back(r);
    }

    sort(bubbles.begin(), bubbles.end(), [](const Rect& a, const Rect& b) {
        if (abs(a.y - b.y) > 8) return a.y < b.y;
        return a.x < b.x;
    });
    return bubbles;
}

Rect findReferenceAvatarForBubble(const Mat& img, const Rect& bubble) {
    int searchRight = max(0, bubble.x - 4);
    int searchLeft = max(0, searchRight - 190);
    int searchTop = max(0, bubble.y - 110);
    int searchBottom = min(img.rows, bubble.y + bubble.height + 30);
    if (searchRight <= searchLeft || searchBottom <= searchTop) return Rect();

    Rect searchRect(searchLeft, searchTop, searchRight - searchLeft, searchBottom - searchTop);
    Mat roi = img(searchRect).clone();

    Scalar bg = estimateBackgroundFromCorners(roi);
    Mat bgMat(roi.size(), roi.type(), bg);
    Mat diffBgr;
    absdiff(roi, bgMat, diffBgr);

    vector<Mat> diffCh;
    split(diffBgr, diffCh);
    Mat maxDiff;
    max(diffCh[0], diffCh[1], maxDiff);
    max(maxDiff, diffCh[2], maxDiff);

    Mat diffMask;
    threshold(maxDiff, diffMask, 14, 255, THRESH_BINARY);

    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 40, 120);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(edges, edges, kernel3);

    Mat combined = diffMask | edges;
    morphologyEx(combined, combined, MORPH_CLOSE, kernel5);
    morphologyEx(combined, combined, MORPH_OPEN, kernel3);

    vector<vector<Point>> contours;
    findContours(combined, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double bestScore = 1e18;
    Rect bestRect;

    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        if (r.area() < 1200) continue;
        if (r.width < 36 || r.height < 36) continue;
        if (r.width > 140 || r.height > 140) continue;

        float ratio = static_cast<float>(r.width) / max(1, r.height);
        if (ratio < 0.75f || ratio > 1.35f) continue;

        Rect global(searchRect.x + r.x, searchRect.y + r.y, r.width, r.height);
        global &= Rect(0, 0, img.cols, img.rows);
        if (global.area() <= 0) continue;

        Mat candidate = img(global);
        Mat candGray;
        cvtColor(candidate, candGray, COLOR_BGR2GRAY);
        Scalar meanGray, stdGray;
        meanStdDev(candGray, meanGray, stdGray);
        if (stdGray[0] < 18.0) continue;

        Mat candEdges;
        Canny(candGray, candEdges, 50, 150);
        double edgeRatio = static_cast<double>(countNonZero(candEdges)) / max(1, global.area());
        if (edgeRatio < 0.010) continue;

        double gapX = abs((global.x + global.width) - bubble.x);
        double offsetY = bubble.y - global.y;
        double score = gapX * 4.0 + abs(offsetY - global.height * 0.55) * 2.0 + abs(global.width - global.height);
        if (score < bestScore) {
            bestScore = score;
            bestRect = global;
        }
    }

    return bestRect;
}

bool detectNicknameRectAboveBubble(const Mat& img,
                                   const Rect& avatarRect,
                                   const Rect& bubbleRect,
                                   Rect& nicknameRect,
                                   Scalar& nicknameColor) {
    nicknameRect = Rect();
    nicknameColor = Scalar(135, 135, 135);
    if (img.empty() || avatarRect.area() <= 0 || bubbleRect.area() <= 0) return false;

    int x1 = min(img.cols - 1, avatarRect.x + avatarRect.width + 4);
    int x2 = min(img.cols, max(bubbleRect.x + min(bubbleRect.width + 80, img.cols / 3), x1 + 24));
    int y1 = max(0, avatarRect.y - avatarRect.height / 10);
    int y2 = min(img.rows, bubbleRect.y - 4);
    if (x2 <= x1 || y2 <= y1) return false;

    Rect roiRect(x1, y1, x2 - x1, y2 - y1);
    Mat roi = img(roiRect).clone();

    Scalar bg = estimateBackgroundFromCorners(roi);
    double bgGray = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2];

    Mat gray, hsv;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    Mat bgMat(roi.size(), roi.type(), bg);
    Mat diffBgr;
    absdiff(roi, bgMat, diffBgr);
    vector<Mat> diffCh;
    split(diffBgr, diffCh);
    Mat maxDiff;
    max(diffCh[0], diffCh[1], maxDiff);
    max(maxDiff, diffCh[2], maxDiff);

    Mat diffMask;
    threshold(maxDiff, diffMask, 8, 255, THRESH_BINARY);

    int grayThr = max(0, static_cast<int>(bgGray - 12.0));
    Mat darkMask;
    threshold(gray, darkMask, grayThr, 255, THRESH_BINARY_INV);

    Mat lowSatMask;
    inRange(hsv, Scalar(0, 0, 0), Scalar(180, 85, 225), lowSatMask);

    Mat mask;
    bitwise_and(darkMask, lowSatMask, mask);
    bitwise_and(mask, diffMask, mask);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel3);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel3);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat mergedMask = Mat::zeros(mask.size(), CV_8UC1);
    Rect unionRect;
    bool found = false;

    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        if (r.area() < 6) continue;
        if (r.width < 2 || r.height < 7) continue;
        if (r.height > roi.rows) continue;

        drawContours(mergedMask, vector<vector<Point>>{c}, -1, Scalar(255), FILLED);
        unionRect = found ? (unionRect | r) : r;
        found = true;
    }

    if (!found || unionRect.width < 10 || unionRect.height < 8) {
        return false;
    }

    nicknameRect = Rect(roiRect.x + unionRect.x,
                        roiRect.y + unionRect.y,
                        unionRect.width,
                        unionRect.height);
    nicknameColor = mean(roi(unionRect), mergedMask(unionRect));
    return true;
}

vector<Rect> findAllAvatars(const Mat& img, const vector<Rect>& bubbles) {
    vector<Rect> avatars;

    for (auto& b : bubbles) {
        int searchX = min(b.x + b.width + 5, img.cols - 1);
        int searchY = max(b.y - 20, 0);
        int searchW = min(180, img.cols - searchX);
        int searchH = min(max(b.height, 140) + 40, img.rows - searchY);

        if (searchW <= 0 || searchH <= 0) continue;

        Rect searchRect(searchX, searchY, searchW, searchH);
        Mat searchROI = img(searchRect).clone();

        Mat gray;
        cvtColor(searchROI, gray, COLOR_BGR2GRAY);

        Mat brightMask;
        threshold(gray, brightMask, 200, 255, THRESH_BINARY);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(brightMask, brightMask, MORPH_OPEN, kernel);
        morphologyEx(brightMask, brightMask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(brightMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto& c : contours) {
            Rect r = boundingRect(c);
            if (r.area() < 3500) continue;
            if (r.width < 60 || r.height < 60) continue;

            Rect globalRect(searchRect.x + r.x, searchRect.y + r.y, r.width, r.height);
            globalRect &= Rect(0, 0, img.cols, img.rows);

            if (globalRect.width <= 0 || globalRect.height <= 0) continue;

            bool duplicate = false;
            for (auto& a : avatars) {
                Rect inter = a & globalRect;
                if (inter.area() > 0.6 * min(a.area(), globalRect.area())) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            Mat candidate = img(globalRect).clone();
            if (looksLikeAvatar(candidate)) {
                avatars.push_back(globalRect);
            }
        }
    }

    sort(avatars.begin(), avatars.end(), [](const Rect& a, const Rect& b) {
        if (abs(a.y - b.y) > 10) return a.y < b.y;
        return a.x < b.x;
    });

    return avatars;
}

Scalar estimateBackgroundFromCorners(const Mat& patch) {
    int cw = max(2, patch.cols / 6);
    int ch = max(2, patch.rows / 6);

    vector<Rect> corners = {
        Rect(0, 0, cw, ch),
        Rect(patch.cols - cw, 0, cw, ch),
        Rect(0, patch.rows - ch, cw, ch),
        Rect(patch.cols - cw, patch.rows - ch, cw, ch)
    };

    Vec3d sum(0, 0, 0);
    int count = 0;

    for (const auto& r : corners) {
        Scalar m = mean(patch(r));
        sum[0] += m[0];
        sum[1] += m[1];
        sum[2] += m[2];
        count++;
    }

    if (count == 0) return Scalar(200, 200, 200);
    return Scalar(sum[0] / count, sum[1] / count, sum[2] / count);
}

Rect bboxFromMaskProjection(const Mat& mask, double colRatio, double rowRatio) {
    if (mask.empty() || mask.type() != CV_8UC1) return Rect();

    int x1 = -1, x2 = -1, y1 = -1, y2 = -1;
    int colThr = max(1, static_cast<int>(mask.rows * colRatio));
    int rowThr = max(1, static_cast<int>(mask.cols * rowRatio));

    for (int x = 0; x < mask.cols; ++x) {
        if (countNonZero(mask.col(x)) >= colThr) {
            x1 = x;
            break;
        }
    }
    for (int x = mask.cols - 1; x >= 0; --x) {
        if (countNonZero(mask.col(x)) >= colThr) {
            x2 = x;
            break;
        }
    }
    for (int y = 0; y < mask.rows; ++y) {
        if (countNonZero(mask.row(y)) >= rowThr) {
            y1 = y;
            break;
        }
    }
    for (int y = mask.rows - 1; y >= 0; --y) {
        if (countNonZero(mask.row(y)) >= rowThr) {
            y2 = y;
            break;
        }
    }

    if (x1 < 0 || x2 < x1 || y1 < 0 || y2 < y1) return Rect();
    return Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

AvatarContentResult detectVisibleAvatarContent(const Mat& patch) {
    AvatarContentResult result;
    result.contentRect = Rect(0, 0, patch.cols, patch.rows);
    result.visibleMask = Mat::zeros(patch.size(), CV_8UC1);

    if (patch.empty()) {
        return result;
    }

    Scalar bg = estimateBackgroundFromCorners(patch);

    Mat bgMat(patch.size(), patch.type(), bg);
    Mat diffBgr;
    absdiff(patch, bgMat, diffBgr);

    vector<Mat> ch;
    split(diffBgr, ch);

    Mat maxDiff;
    max(ch[0], ch[1], maxDiff);
    max(maxDiff, ch[2], maxDiff);

    Mat diffMask;
    threshold(maxDiff, diffMask, 14, 255, THRESH_BINARY);

    Mat gray;
    cvtColor(patch, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, 40, 120);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));

    dilate(edges, edges, kernel3);
    Mat combined = diffMask | edges;

    morphologyEx(combined, combined, MORPH_CLOSE, kernel5);
    morphologyEx(combined, combined, MORPH_OPEN, kernel3);

    Rect projRect = bboxFromMaskProjection(combined, 0.06, 0.06);

    vector<vector<Point>> contours;
    findContours(combined, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double bestScore = -1e18;
    int bestIdx = -1;
    Point2f center(static_cast<float>(patch.cols) / 2.0f, static_cast<float>(patch.rows) / 2.0f);

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < patch.total() * 0.03) continue;

        Rect r = boundingRect(contours[i]);
        Point2f rc(static_cast<float>(r.x + r.width / 2.0f), static_cast<float>(r.y + r.height / 2.0f));
        double dist = norm(rc - center);

        double score = area - 0.15 * dist * dist;
        if (score > bestScore) {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        }
    }

    Rect contourRect;
    Mat contourMask = Mat::zeros(patch.size(), CV_8UC1);

    if (bestIdx >= 0) {
        drawContours(contourMask, contours, bestIdx, Scalar(255), FILLED);
        morphologyEx(contourMask, contourMask, MORPH_CLOSE, kernel5);
        contourRect = boundingRect(contours[bestIdx]);
    }

    Rect finalRect;
    if (contourRect.area() > 0 && projRect.area() > 0) {
        int x1 = min(contourRect.x, projRect.x);
        int y1 = min(contourRect.y, projRect.y);
        int x2 = max(contourRect.x + contourRect.width, projRect.x + projRect.width);
        int y2 = max(contourRect.y + contourRect.height, projRect.y + projRect.height);
        finalRect = Rect(x1, y1, x2 - x1, y2 - y1);
    } else if (contourRect.area() > 0) {
        finalRect = contourRect;
    } else if (projRect.area() > 0) {
        finalRect = projRect;
        rectangle(contourMask, finalRect, Scalar(255), FILLED);
    } else {
        finalRect = Rect(0, 0, patch.cols, patch.rows);
        contourMask.setTo(255);
    }

    finalRect = expandRect(finalRect, 2, patch.size());

    if (finalRect.area() < patch.total() * 0.15) {
        finalRect = Rect(0, 0, patch.cols, patch.rows);
        contourMask.setTo(255);
    }

    Mat solidMask = Mat::zeros(patch.size(), CV_8UC1);
    rectangle(solidMask, finalRect, Scalar(255), FILLED);

    result.contentRect = finalRect;
    result.visibleMask = solidMask;
    return result;
}

vector<AvatarInfo> buildAvatarInfos(const Mat& img, const vector<Rect>& avatarRects) {
    vector<AvatarInfo> infos;

    for (const auto& r : avatarRects) {
        if (r.width <= 0 || r.height <= 0) continue;
        Mat patch = img(r).clone();

        AvatarContentResult det = detectVisibleAvatarContent(patch);

        AvatarInfo info;
        info.outerRect = r;
        info.contentRectInPatch = det.contentRect;
        info.patch = patch;
        info.visibleMask = det.visibleMask;
        infos.push_back(info);
    }

    return infos;
}

Rect contentRectInImage(const AvatarInfo& info) {
    return Rect(
        info.outerRect.x + info.contentRectInPatch.x,
        info.outerRect.y + info.contentRectInPatch.y,
        info.contentRectInPatch.width,
        info.contentRectInPatch.height
    );
}

Rect contentRectInTargetImage(const AvatarInfo& info, const Rect& targetOuter) {
    return Rect(
        targetOuter.x + info.contentRectInPatch.x,
        targetOuter.y + info.contentRectInPatch.y,
        info.contentRectInPatch.width,
        info.contentRectInPatch.height
    );
}

Rect detectAvatarContentRectInImage(const Mat& img, const Rect& avatarOuterRect) {
    if (img.empty() || avatarOuterRect.area() <= 0) return avatarOuterRect;
    AvatarContentResult det = detectVisibleAvatarContent(img(avatarOuterRect).clone());
    Rect local = det.contentRect.area() > 0
        ? det.contentRect
        : Rect(0, 0, avatarOuterRect.width, avatarOuterRect.height);
    return Rect(avatarOuterRect.x + local.x, avatarOuterRect.y + local.y, local.width, local.height);
}

Rect mirroredOuterRectByContent(const AvatarInfo& info, int centerX) {
    Rect contentGlobal = contentRectInImage(info);

    int targetContentX = 2 * centerX - (contentGlobal.x + contentGlobal.width);
    int targetOuterX = targetContentX - info.contentRectInPatch.x;

    return Rect(targetOuterX, info.outerRect.y, info.outerRect.width, info.outerRect.height);
}

NicknameReferenceStyle estimateNicknameReferenceStyle(const Mat& img,
                                                      const vector<AvatarInfo>& avatarInfos,
                                                      const vector<BubbleInfo>& bubbleInfos) {
    NicknameReferenceStyle style;
    style.cjkFontPath = pickNicknameCjkFontPath();
    style.cjkFontIndex = fontFaceIndexForNicknamePath(style.cjkFontPath);
    style.latinFontPath = style.cjkFontPath;
    style.latinFontIndex = style.cjkFontIndex;

    vector<int> avatarSizes;
    vector<int> bubbleTopOffsets;
    vector<int> nicknameTopOffsets;
    vector<int> bubbleXs;
    vector<int> nicknameXs;
    vector<Rect> referenceNicknameRects;
    vector<Rect> leftWhiteBubbles = findLeftWhiteBubbles(img);

    for (const auto& bubbleRect : leftWhiteBubbles) {
        Mat whiteBubbleMask = extractWhiteBubbleMask(img, bubbleRect);
        int bubbleTipX = leftmostMaskXInBand(whiteBubbleMask);
        bubbleXs.push_back(bubbleTipX >= 0 ? (bubbleRect.x - 3 + bubbleTipX) : bubbleRect.x);

        Rect avatarRect = findReferenceAvatarForBubble(img, bubbleRect);
        if (avatarRect.area() <= 0) continue;
        Rect avatarContentRect = detectAvatarContentRectInImage(img, avatarRect);
        if (avatarContentRect.area() <= 0) continue;

        avatarSizes.push_back(min(avatarContentRect.width, avatarContentRect.height));
        bubbleTopOffsets.push_back(bubbleRect.y - avatarContentRect.y);

        Rect nicknameRect;
        Scalar nicknameColor;
        if (detectNicknameRectAboveBubble(img, avatarRect, bubbleRect, nicknameRect, nicknameColor)) {
            referenceNicknameRects.push_back(nicknameRect);
            nicknameTopOffsets.push_back(nicknameRect.y - avatarRect.y);
            nicknameXs.push_back(nicknameRect.x);
        }
    }

    int avatarSize = medianInt(avatarSizes, 95);
    float scale = static_cast<float>(avatarSize) / 95.0f;

    style.valid = true;
    style.color = Scalar(133, 133, 133);
    style.referenceAvatarSize = avatarSize;
    int baseFontHeight = max(18, min(30, static_cast<int>(std::round(24.0f * scale))));
    style.fontHeight = max(18, static_cast<int>(std::round(baseFontHeight * NICKNAME_FONT_SCALE)));
    style.latinFontHeight = style.fontHeight;
    style.textXOffsetFromAvatarRight = max(8, static_cast<int>(std::round(29.0f * scale)));
    style.textTopOffsetFromAvatarTop = medianInt(
        nicknameTopOffsets,
        static_cast<int>(std::round(-4.0f * scale))
    ) - max(1, style.fontHeight / 8);
    style.bubbleXOffsetFromAvatarRight = max(8, static_cast<int>(std::round(17.0f * scale)));
    style.nicknameToBubbleGap = max(2, static_cast<int>(std::round(4.0f * scale)));
    style.bubbleYOffsetFromAvatarTop = medianInt(
        bubbleTopOffsets,
        static_cast<int>(std::round(39.0f * scale))
    );
    style.bubbleYOffsetFromAvatarTop = max(
        style.bubbleYOffsetFromAvatarTop,
        style.textTopOffsetFromAvatarTop + style.fontHeight + style.nicknameToBubbleGap + 2
    );
    style.blockGapY = max(6, static_cast<int>(std::round(10.0f * scale)));
    style.referenceBubbleX = medianInt(bubbleXs, medianRectX(leftWhiteBubbles, -1));
    style.referenceNicknameX = medianInt(nicknameXs, medianRectX(referenceNicknameRects, style.referenceBubbleX));

    return style;
}

struct NicknameTextRenderer {
    Ptr<freetype::FreeType2> latin;
    Ptr<freetype::FreeType2> cjk;
};

Ptr<freetype::FreeType2> createSingleFontRenderer(const string& fontPath, int faceIndex) {
    if (fontPath.empty()) return nullptr;
    try {
        auto ft2 = freetype::createFreeType2();
        ft2->loadFontData(fontPath, faceIndex);
        ft2->setSplitNumber(8);
        return ft2;
    } catch (const cv::Exception& e) {
        cerr << "Failed to load nickname font: " << fontPath << " (" << e.what() << ")" << endl;
        return nullptr;
    }
}

NicknameTextRenderer createNicknameTextRenderer(const NicknameReferenceStyle& style) {
    NicknameTextRenderer renderer;
    renderer.latin = createSingleFontRenderer(style.latinFontPath, style.latinFontIndex);
    renderer.cjk = createSingleFontRenderer(style.cjkFontPath, style.cjkFontIndex);
    return renderer;
}

Ptr<freetype::FreeType2> nicknameRunRenderer(bool asciiRun, const NicknameTextRenderer& renderer) {
    Ptr<freetype::FreeType2> ft = asciiRun ? renderer.latin : renderer.cjk;
    if (!ft) ft = asciiRun ? renderer.cjk : renderer.latin;
    return ft;
}

int nicknameRunFontHeight(const NicknameReferenceStyle& style, bool asciiRun) {
    return asciiRun ? max(1, style.latinFontHeight) : max(1, style.fontHeight);
}

double nicknameFallbackScale(int fontHeight) {
    return max(0.45, fontHeight / 28.0);
}

Size measureNicknameRunAtHeight(const string& text,
                                bool asciiRun,
                                int fontHeight,
                                const NicknameTextRenderer& renderer,
                                int* baseline) {
    int safeBaseline = 0;
    int* baselinePtr = baseline ? baseline : &safeBaseline;
    Ptr<freetype::FreeType2> ft = nicknameRunRenderer(asciiRun, renderer);

    if (ft) {
        return ft->getTextSize(text, fontHeight, -1, baselinePtr);
    }

    return getTextSize(text, FONT_HERSHEY_SIMPLEX,
                       nicknameFallbackScale(fontHeight), 1, baselinePtr);
}

int measureNicknameInkHeight(const string& text,
                             bool asciiRun,
                             int fontHeight,
                             const NicknameTextRenderer& renderer) {
    if (text.empty()) return 0;

    int baseline = 0;
    Size runSize = measureNicknameRunAtHeight(text, asciiRun, fontHeight, renderer, &baseline);
    if (runSize.width <= 0 || runSize.height <= 0) return 0;

    Mat canvas(runSize.height + baseline + 24, runSize.width + 24, CV_8UC3, Scalar(0, 0, 0));
    Point origin(12, 12 + runSize.height);
    Ptr<freetype::FreeType2> ft = nicknameRunRenderer(asciiRun, renderer);
    if (ft) {
        ft->putText(canvas, text, origin, fontHeight, Scalar(255, 255, 255), -1, LINE_AA, false);
    } else {
        putText(canvas, text, origin, FONT_HERSHEY_SIMPLEX,
                nicknameFallbackScale(fontHeight), Scalar(255, 255, 255), 1, LINE_AA);
    }

    Mat mask;
    cvtColor(canvas, mask, COLOR_BGR2GRAY);
    Mat binary;
    threshold(mask, binary, 1, 255, THRESH_BINARY);
    vector<Point> nonZeroPts;
    findNonZero(binary, nonZeroPts);
    if (nonZeroPts.empty()) return 0;
    return boundingRect(nonZeroPts).height;
}

vector<pair<string, bool>> splitNicknameRuns(const string& text);
Size measureNicknameRun(const string& text,
                        bool asciiRun,
                        const NicknameReferenceStyle& style,
                        const NicknameTextRenderer& renderer,
                        int* baseline);
Size measureNicknameText(const string& text,
                         const NicknameReferenceStyle& style,
                         const NicknameTextRenderer& renderer,
                         int* baseline);

Rect measureNicknameInkBounds(const string& text,
                              const NicknameReferenceStyle& style,
                              const NicknameTextRenderer& renderer,
                              bool lowercaseOnly,
                              int alphaThreshold) {
    if (text.empty()) return Rect();

    int baseline = 0;
    Size textSize = measureNicknameText(text, style, renderer, &baseline);
    if (textSize.width <= 0 || textSize.height <= 0) return Rect();

    const int pad = 20;
    Mat canvas(textSize.height + baseline + pad * 2, textSize.width + pad * 2, CV_8UC3, Scalar(0, 0, 0));
    int cursorX = pad;
    int baselineY = pad + textSize.height;

    for (const auto& run : splitNicknameRuns(text)) {
        int runBaseline = 0;
        Size runSize = measureNicknameRun(run.first, run.second, style, renderer, &runBaseline);
        Ptr<freetype::FreeType2> ft = nicknameRunRenderer(run.second, renderer);
        int runFontHeight = nicknameRunFontHeight(style, run.second);
        string drawText = run.first;

        if (lowercaseOnly) {
            if (!run.second) {
                cursorX += runSize.width;
                continue;
            }
            bool hasLowercase = false;
            for (char& ch : drawText) {
                bool keep = ch >= 'a' && ch <= 'z';
                hasLowercase = hasLowercase || keep;
                if (!keep) ch = ' ';
            }
            if (!hasLowercase) {
                cursorX += runSize.width;
                continue;
            }
        }

        Point origin(cursorX, baselineY);
        if (ft) {
            ft->putText(canvas, drawText, origin, runFontHeight, Scalar(255, 255, 255), -1, LINE_AA, false);
        } else {
            putText(canvas, drawText, origin, FONT_HERSHEY_SIMPLEX,
                    nicknameFallbackScale(runFontHeight), Scalar(255, 255, 255), 1, LINE_AA);
        }
        cursorX += runSize.width;
    }

    Mat mask;
    cvtColor(canvas, mask, COLOR_BGR2GRAY);
    Mat strongMask;
    threshold(mask, strongMask, alphaThreshold, 255, THRESH_BINARY);
    Rect bounds = boundingRectOfMask(strongMask);
    if (bounds.area() <= 0) {
        bounds = boundingRectOfMask(mask);
    }
    if (bounds.area() <= 0) return Rect();
    bounds.x -= pad;
    bounds.y -= pad;
    return bounds;
}

int calibrateNicknameLatinFontHeight(const NicknameReferenceStyle& style,
                                     const NicknameTextRenderer& renderer) {
    int baseHeight = max(style.fontHeight, 24);
    float avatarScale = max(1.0f, static_cast<float>(style.referenceAvatarSize) / 95.0f);
    int targetInkHeight = max(21, static_cast<int>(std::round(22.0f * avatarScale)));

    int bestHeight = baseHeight;
    int bestDiff = numeric_limits<int>::max();
    bool foundAtOrAboveTarget = false;

    for (int fontHeight = max(24, baseHeight - 2); fontHeight <= 52; ++fontHeight) {
        int inkHeight = measureNicknameInkHeight("en", true, fontHeight, renderer);
        if (inkHeight <= 0) continue;

        int diff = abs(inkHeight - targetInkHeight);
        if (inkHeight >= targetInkHeight) {
            if (!foundAtOrAboveTarget || fontHeight < bestHeight ||
                (fontHeight == bestHeight && diff < bestDiff)) {
                bestHeight = fontHeight;
                bestDiff = diff;
                foundAtOrAboveTarget = true;
            }
        } else if (!foundAtOrAboveTarget && diff < bestDiff) {
            bestHeight = fontHeight;
            bestDiff = diff;
        }
    }

    return max(baseHeight, bestHeight);
}

void calibrateNicknameStyleHeights(NicknameReferenceStyle& style,
                                   const NicknameTextRenderer& renderer) {
    style.latinFontHeight = calibrateNicknameLatinFontHeight(style, renderer);
}

vector<pair<string, bool>> splitNicknameRuns(const string& text) {
    vector<pair<string, bool>> runs;
    if (text.empty()) return runs;

    size_t start = 0;
    bool currentAscii = static_cast<unsigned char>(text[0]) < 0x80;
    for (size_t i = 1; i < text.size(); ++i) {
        bool ascii = static_cast<unsigned char>(text[i]) < 0x80;
        if (ascii != currentAscii) {
            runs.push_back({text.substr(start, i - start), currentAscii});
            start = i;
            currentAscii = ascii;
        }
    }
    runs.push_back({text.substr(start), currentAscii});
    return runs;
}

Size measureNicknameRun(const string& text,
                        bool asciiRun,
                        const NicknameReferenceStyle& style,
                        const NicknameTextRenderer& renderer,
                        int* baseline) {
    return measureNicknameRunAtHeight(
        text, asciiRun, nicknameRunFontHeight(style, asciiRun), renderer, baseline
    );
}

Size measureNicknameText(const string& text,
                         const NicknameReferenceStyle& style,
                         const NicknameTextRenderer& renderer,
                         int* baseline) {
    int totalWidth = 0;
    int maxHeight = 0;
    int maxBaseline = 0;

    for (const auto& run : splitNicknameRuns(text)) {
        int runBaseline = 0;
        Size runSize = measureNicknameRun(run.first, run.second, style, renderer, &runBaseline);
        totalWidth += runSize.width;
        maxHeight = max(maxHeight, runSize.height);
        maxBaseline = max(maxBaseline, runBaseline);
    }

    if (baseline) *baseline = maxBaseline;
    return Size(totalWidth, maxHeight);
}

void drawNicknameText(Mat& img,
                      const string& text,
                      const Rect& textRect,
                      const NicknameReferenceStyle& style,
                      const NicknameTextRenderer& renderer) {
    if (img.empty() || text.empty() || textRect.area() <= 0) return;

    int baseline = 0;
    Size textSize = measureNicknameText(text, style, renderer, &baseline);
    int cursorX = textRect.x;
    int baselineY = textRect.y + textSize.height;

    for (const auto& run : splitNicknameRuns(text)) {
        int runBaseline = 0;
        Size runSize = measureNicknameRun(run.first, run.second, style, renderer, &runBaseline);
        Ptr<freetype::FreeType2> ft = nicknameRunRenderer(run.second, renderer);
        int runFontHeight = nicknameRunFontHeight(style, run.second);

        Point origin(cursorX, baselineY);
        if (ft) {
            ft->putText(img, run.first, origin, runFontHeight, style.color, -1, LINE_AA, false);
        } else {
            double fontScale = nicknameFallbackScale(runFontHeight);
            putText(img, run.first, origin, FONT_HERSHEY_SIMPLEX, fontScale, style.color, 1, LINE_AA);
        }
        cursorX += runSize.width;
    }
}

// ========================
// 绿色聊天框相关
// ========================

Mat buildSolidBubbleMask(const Mat& patch) {
    if (patch.empty()) return Mat();

    Mat hsv;
    cvtColor(patch, hsv, COLOR_BGR2HSV);

    Scalar lower(35, 60, 120);
    Scalar upper(70, 255, 255);

    Mat greenMask;
    inRange(hsv, lower, upper, greenMask);

    Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    morphologyEx(greenMask, greenMask, MORPH_OPEN, kernel3);
    morphologyEx(greenMask, greenMask, MORPH_CLOSE, kernel5);

    vector<vector<Point>> contours;
    findContours(greenMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat solidMask = Mat::zeros(patch.size(), CV_8UC1);

    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area < patch.total() * 0.02) continue;
        drawContours(solidMask, vector<vector<Point>>{c}, -1, Scalar(255), FILLED);
    }

    morphologyEx(solidMask, solidMask, MORPH_CLOSE, kernel5);

    if (countNonZero(solidMask) == 0) {
        rectangle(solidMask, Rect(0, 0, patch.cols, patch.rows), Scalar(255), FILLED);
    }

    return solidMask;
}

Mat buildBubbleInteriorMask(const Mat& bubbleMask) {
    if (bubbleMask.empty()) return Mat();

    Mat interiorMask = bubbleMask.clone();
    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(interiorMask, interiorMask, kernel3);
    erode(interiorMask, interiorMask, kernel5);
    if (countNonZero(interiorMask) == 0) {
        interiorMask = bubbleMask.clone();
    }
    return interiorMask;
}

Scalar estimateBubbleFillColor(const Mat& patch, const Mat& bubbleMask) {
    if (patch.empty() || bubbleMask.empty()) return Scalar(120, 230, 80);

    Mat interiorMask = buildBubbleInteriorMask(bubbleMask);
    Scalar fillColor = mean(patch, interiorMask);
    return Scalar(fillColor[0], fillColor[1], fillColor[2]);
}

Scalar estimateReferenceWhiteBubbleColor(const Mat& img) {
    if (img.empty()) return DEFAULT_MIRRORED_BUBBLE_FILL;

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    Mat whiteMask;
    inRange(hsv, Scalar(0, 0, 215), Scalar(180, 40, 255), whiteMask);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(whiteMask, whiteMask, MORPH_OPEN, kernel3);
    morphologyEx(whiteMask, whiteMask, MORPH_CLOSE, kernel5);

    vector<vector<Point>> contours;
    findContours(whiteMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Vec3d weightedSum(0, 0, 0);
    double totalWeight = 0.0;

    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        if (r.area() < 1200) continue;
        if (r.area() > img.cols * img.rows * 0.08) continue;
        if (r.width < 45 || r.height < 28) continue;
        if (r.width > img.cols * 0.55) continue;
        if (r.x > img.cols * 0.60) continue;
        if (r.y < img.rows * 0.08 || r.y > img.rows * 0.92) continue;

        float ratio = static_cast<float>(r.width) / max(1, r.height);
        if (ratio < 0.8f || ratio > 8.5f) continue;

        Mat contourMask = Mat::zeros(img.size(), CV_8UC1);
        drawContours(contourMask, vector<vector<Point>>{c}, -1, Scalar(255), FILLED);

        Scalar sample = mean(img, contourMask);
        double weight = contourArea(c);
        weightedSum[0] += sample[0] * weight;
        weightedSum[1] += sample[1] * weight;
        weightedSum[2] += sample[2] * weight;
        totalWeight += weight;
    }

    if (totalWeight <= 0.0) return DEFAULT_MIRRORED_BUBBLE_FILL;

    return Scalar(weightedSum[0] / totalWeight,
                  weightedSum[1] / totalWeight,
                  weightedSum[2] / totalWeight);
}

BubbleContentResult detectBubbleContent(const Mat& patch, const Mat& bubbleMask, const Scalar& fillColor) {
    BubbleContentResult result;
    result.bodyRect = Rect();
    result.contentRect = Rect();

    if (patch.empty() || bubbleMask.empty()) return result;

    Mat interiorMask = buildBubbleInteriorMask(bubbleMask);
    Rect bodyRect = bboxFromMaskPeakThreshold(interiorMask, 0.82, 0.82);
    if (bodyRect.area() == 0) {
        bodyRect = bboxFromMaskProjection(interiorMask, 0.08, 0.08);
    }
    if (bodyRect.area() == 0) {
        bodyRect = Rect(0, 0, patch.cols, patch.rows);
    }

    bodyRect = insetRect(bodyRect, 1, 1, patch.size());
    Rect searchRect = insetRect(bodyRect, 2, 2, patch.size());
    if (searchRect.area() == 0) {
        searchRect = bodyRect;
    }

    Mat fillMat(patch.size(), patch.type(), fillColor);
    Mat diffBgr;
    absdiff(patch, fillMat, diffBgr);

    vector<Mat> channels;
    split(diffBgr, channels);
    Mat maxDiff;
    max(channels[0], channels[1], maxDiff);
    max(maxDiff, channels[2], maxDiff);

    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));

    Mat bodyMask = Mat::zeros(patch.size(), CV_8UC1);
    rectangle(bodyMask, bodyRect, Scalar(255), FILLED);
    bitwise_and(bodyMask, interiorMask, bodyMask);

    Mat searchMask = Mat::zeros(patch.size(), CV_8UC1);
    rectangle(searchMask, searchRect, Scalar(255), FILLED);
    bitwise_and(searchMask, bodyMask, searchMask);

    Mat looseMask;
    threshold(maxDiff, looseMask, 8, 255, THRESH_BINARY);
    bitwise_and(looseMask, searchMask, looseMask);

    Mat coreMask;
    threshold(maxDiff, coreMask, 18, 255, THRESH_BINARY);
    bitwise_and(coreMask, searchMask, coreMask);
    morphologyEx(coreMask, coreMask, MORPH_CLOSE, kernel3);

    Mat expandedCore;
    dilate(coreMask, expandedCore, kernel3);

    Mat rawMask;
    bitwise_and(looseMask, expandedCore, rawMask);
    morphologyEx(rawMask, rawMask, MORPH_CLOSE, kernel3);

    vector<vector<Point>> contours;
    findContours(coreMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat cleanedMask = Mat::zeros(patch.size(), CV_8UC1);
    Rect unionRect;
    bool hasContent = false;

    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        double area = contourArea(c);
        if (area < 6) continue;
        if (r.area() < 16) continue;
        if (r.width < 2 || r.height < 4) continue;

        Rect expanded = expandRect(r, 2, patch.size());
        Mat componentMask = Mat::zeros(patch.size(), CV_8UC1);
        rawMask(expanded).copyTo(componentMask(expanded));

        bitwise_and(componentMask, searchMask, componentMask);
        if (countNonZero(componentMask) == 0) continue;

        bitwise_or(cleanedMask, componentMask, cleanedMask);
        unionRect = hasContent ? (unionRect | expanded) : expanded;
        hasContent = true;
    }

    if (!hasContent) {
        result.bodyRect = bodyRect;
        result.bodyMaskFull = bodyMask;
        result.searchMaskFull = searchMask;
        result.coreMaskFull = coreMask;
        result.looseMaskFull = looseMask;
        return result;
    }

    Mat eraseMask;
    dilate(cleanedMask, eraseMask, kernel3);
    morphologyEx(eraseMask, eraseMask, MORPH_CLOSE, kernel5);
    bitwise_and(eraseMask, searchMask, eraseMask);

    result.contentRect = unionRect;
    result.visibleMask = cleanedMask(unionRect).clone();
    result.bodyRect = bodyRect;
    result.pasteMaskFull = cleanedMask;
    result.eraseMaskFull = eraseMask;
    result.bodyMaskFull = bodyMask;
    result.searchMaskFull = searchMask;
    result.coreMaskFull = coreMask;
    result.looseMaskFull = looseMask;
    return result;
}

vector<BubbleInfo> buildBubbleInfos(const Mat& img, const vector<Rect>& bubbleRects) {
    vector<BubbleInfo> infos;

    for (auto r : bubbleRects) {
        r = expandRect(r, 2, img.size());
        if (r.width <= 0 || r.height <= 0) continue;

        BubbleInfo info;
        info.outerRect = r;
        info.patch = img(r).clone();
        info.visibleMask = buildSolidBubbleMask(info.patch);
        info.fillColor = estimateBubbleFillColor(info.patch, info.visibleMask);

        BubbleContentResult content = detectBubbleContent(info.patch, info.visibleMask, info.fillColor);
        info.bodyRectInPatch = content.bodyRect;
        info.bodyMaskFull = content.bodyMaskFull.clone();
        info.searchMaskFull = content.searchMaskFull.clone();
        info.coreMaskFull = content.coreMaskFull.clone();
        info.looseMaskFull = content.looseMaskFull.clone();
        info.pasteMaskFull = content.pasteMaskFull.clone();
        info.eraseMaskFull = content.eraseMaskFull.clone();

        if (content.contentRect.area() > 0 && !content.visibleMask.empty()) {
            info.contentRectInPatch = content.contentRect;
            info.contentPatch = info.patch(content.contentRect).clone();
            info.contentMask = content.visibleMask.clone();
            info.hasContent = true;
        }

        infos.push_back(info);
    }

    return infos;
}

Rect mirroredBubbleRect(const BubbleInfo& info, int centerX) {
    int targetX = 2 * centerX - (info.outerRect.x + info.outerRect.width);
    return Rect(targetX, info.outerRect.y, info.outerRect.width, info.outerRect.height);
}

Rect mirroredRectInPatch(const Rect& r, int patchWidth) {
    return Rect(patchWidth - (r.x + r.width), r.y, r.width, r.height);
}

Rect readableMirroredContentRect(const BubbleInfo& info) {
    Rect mirroredContent = mirroredRectInPatch(info.contentRectInPatch, info.patch.cols);
    if (info.bodyRectInPatch.area() <= 0 || info.contentRectInPatch.area() <= 0) {
        return mirroredContent;
    }

    Rect mirroredBody = mirroredRectInPatch(info.bodyRectInPatch, info.patch.cols);
    int leftPadding = max(0, info.contentRectInPatch.x - info.bodyRectInPatch.x);
    int topPadding = max(0, info.contentRectInPatch.y - info.bodyRectInPatch.y);

    int targetX = mirroredBody.x + leftPadding;
    int targetY = mirroredBody.y + topPadding;

    int maxX = max(0, mirroredBody.x + mirroredBody.width - info.contentRectInPatch.width);
    int maxY = max(0, mirroredBody.y + mirroredBody.height - info.contentRectInPatch.height);
    targetX = min(max(targetX, 0), maxX);
    targetY = min(max(targetY, 0), maxY);

    return Rect(targetX,
                targetY,
                info.contentRectInPatch.width,
                info.contentRectInPatch.height);
}

Mat dilateMask(const Mat& mask, int ksize) {
    if (mask.empty()) return Mat();
    int size = max(1, ksize | 1);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(size, size));
    Mat out;
    dilate(mask, out, kernel);
    return out;
}

Mat buildBubbleCleanupMask(const BubbleInfo& info) {
    if (info.visibleMask.empty()) return Mat();

    Mat cleanup = info.visibleMask.clone();
    int baseSize = max(9, ((min(info.outerRect.width, info.outerRect.height) / 4) | 1));
    cleanup = dilateMask(cleanup, baseSize);

    if (!info.bodyMaskFull.empty()) {
        Mat invBody;
        bitwise_not(info.bodyMaskFull, invBody);

        Mat tailMask;
        bitwise_and(info.visibleMask, invBody, tailMask);

        if (countNonZero(tailMask) > 0) {
            int tailW = max(11, ((info.outerRect.height / 2) | 1));
            int tailH = max(9, ((info.outerRect.height / 3) | 1));
            Mat tailKernel = getStructuringElement(MORPH_ELLIPSE, Size(tailW, tailH));
            dilate(tailMask, tailMask, tailKernel);
            bitwise_or(cleanup, tailMask, cleanup);
        }
    }

    int edgeW = max(8, info.outerRect.height / 4);
    int edgeH = max(12, info.outerRect.height / 2);
    Rect edgeBoost(max(0, cleanup.cols - edgeW - 1),
                   max(0, cleanup.rows / 2 - edgeH / 2),
                   min(edgeW, cleanup.cols),
                   min(edgeH, cleanup.rows - max(0, cleanup.rows / 2 - edgeH / 2)));
    if (edgeBoost.area() > 0) {
        rectangle(cleanup, edgeBoost, Scalar(255), FILLED);
    }

    Mat closeKernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(cleanup, cleanup, MORPH_CLOSE, closeKernel);
    return cleanup;
}

PlacedMask buildPlacedBubbleCleanupMask(const BubbleInfo& info, const Size& imageSize) {
    PlacedMask placed;
    if (info.visibleMask.empty() || info.outerRect.area() <= 0) return placed;

    int pad = max(12, max(info.outerRect.width, info.outerRect.height) / 6);
    placed.rect = expandRect(info.outerRect, pad, imageSize);
    if (placed.rect.area() <= 0) return placed;

    placed.mask = Mat::zeros(placed.rect.size(), CV_8UC1);

    Mat cleanup = buildBubbleCleanupMask(info);
    if (cleanup.empty()) return placed;

    Rect localRect(info.outerRect.x - placed.rect.x,
                   info.outerRect.y - placed.rect.y,
                   info.outerRect.width,
                   info.outerRect.height);
    cleanup.copyTo(placed.mask(localRect));

    int spillW = max(9, ((info.outerRect.height / 3) | 1));
    int spillH = max(9, ((info.outerRect.height / 4) | 1));
    Mat spillKernel = getStructuringElement(MORPH_ELLIPSE, Size(spillW, spillH));
    dilate(placed.mask, placed.mask, spillKernel);

    return placed;
}

Mat adaptTextPatchToTargetFill(const BubbleInfo& info, const Scalar& targetFillColor) {
    if (info.contentPatch.empty() || info.contentMask.empty()) return Mat();

    Mat adjusted = info.contentPatch.clone();

    for (int y = 0; y < adjusted.rows; ++y) {
        const uchar* maskRow = info.contentMask.ptr<uchar>(y);
        const Vec3b* srcRow = info.contentPatch.ptr<Vec3b>(y);
        Vec3b* dstRow = adjusted.ptr<Vec3b>(y);

        for (int x = 0; x < adjusted.cols; ++x) {
            if (!maskRow[x]) continue;

            float alpha = 0.0f;
            for (int c = 0; c < 3; ++c) {
                float srcFill = static_cast<float>(info.fillColor[c]);
                float srcPixel = static_cast<float>(srcRow[x][c]);
                float denom = max(1.0f, srcFill);
                float channelAlpha = (srcFill - srcPixel) / denom;
                alpha = max(alpha, min(1.0f, max(0.0f, channelAlpha)));
            }

            for (int c = 0; c < 3; ++c) {
                float target = static_cast<float>(targetFillColor[c]);
                dstRow[x][c] = saturate_cast<uchar>((1.0f - alpha) * target);
            }
        }
    }

    return adjusted;
}

Mat adaptBubblePatchToTargetFill(const Mat& patch,
                                 const Mat& bubbleMask,
                                 const Scalar& sourceFillColor,
                                 const Scalar& targetFillColor) {
    if (patch.empty()) return Mat();

    Mat adjusted = patch.clone();
    double sourceLuma =
        0.114 * sourceFillColor[0] +
        0.587 * sourceFillColor[1] +
        0.299 * sourceFillColor[2];

    for (int y = 0; y < adjusted.rows; ++y) {
        const uchar* maskRow = bubbleMask.empty() ? nullptr : bubbleMask.ptr<uchar>(y);
        const Vec3b* srcRow = patch.ptr<Vec3b>(y);
        Vec3b* dstRow = adjusted.ptr<Vec3b>(y);

        for (int x = 0; x < adjusted.cols; ++x) {
            if (maskRow && !maskRow[x]) continue;

            double srcLuma =
                0.114 * srcRow[x][0] +
                0.587 * srcRow[x][1] +
                0.299 * srcRow[x][2];
            double shadeDelta = srcLuma - sourceLuma;

            for (int c = 0; c < 3; ++c) {
                double target = targetFillColor[c] + shadeDelta;
                dstRow[x][c] = saturate_cast<uchar>(target);
            }
        }
    }

    return adjusted;
}

Scalar bubbleBorderColor(const Scalar& fillColor) {
    return Scalar(
        max(0.0, fillColor[0] - 12.0),
        max(0.0, fillColor[1] - 12.0),
        max(0.0, fillColor[2] - 12.0)
    );
}

Mat renderCrispBubbleBase(const Mat& bubbleMask, const Scalar& fillColor) {
    if (bubbleMask.empty()) return Mat();

    Mat cleanedMask = bubbleMask.clone();
    Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernel5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(cleanedMask, cleanedMask, MORPH_OPEN, kernel3);
    morphologyEx(cleanedMask, cleanedMask, MORPH_CLOSE, kernel5);

    vector<vector<Point>> contours;
    findContours(cleanedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat solidMask = Mat::zeros(cleanedMask.size(), CV_8UC1);
    for (const auto& c : contours) {
        if (contourArea(c) < cleanedMask.total() * 0.02) continue;
        drawContours(solidMask, vector<vector<Point>>{c}, -1, Scalar(255), FILLED, LINE_8);
    }
    if (countNonZero(solidMask) == 0) {
        solidMask = cleanedMask;
    }

    Mat bubble(solidMask.size(), CV_8UC3, Scalar(0, 0, 0));
    bubble.setTo(fillColor, solidMask);

    Mat innerMask;
    erode(solidMask, innerMask, kernel3);
    Mat borderMask;
    subtract(solidMask, innerMask, borderMask);
    if (countNonZero(borderMask) > 0) {
        bubble.setTo(bubbleBorderColor(fillColor), borderMask);
    }

    return bubble;
}

Mat buildMirroredBubblePatch(const BubbleInfo& info,
                             bool keepTextReadable,
                             const Scalar& targetFillColor) {
    Mat flippedPatch;
    flip(info.patch, flippedPatch, 1);

    Mat flippedBubbleMask;
    flip(info.visibleMask, flippedBubbleMask, 1);
    if (!flippedBubbleMask.empty()) {
        Mat crispBubble = renderCrispBubbleBase(flippedBubbleMask, targetFillColor);
        if (!crispBubble.empty()) {
            flippedPatch = crispBubble;
        } else {
            Mat recoloredBubble = adaptBubblePatchToTargetFill(
                flippedPatch, flippedBubbleMask, info.fillColor, targetFillColor
            );
            if (!recoloredBubble.empty()) {
                flippedPatch = recoloredBubble;
            }
        }
    }

    if (!keepTextReadable || !info.hasContent || info.contentPatch.empty() || info.contentMask.empty()) {
        return flippedPatch;
    }

    Rect mirroredContent = readableMirroredContentRect(info);
    if ((mirroredContent & Rect(0, 0, flippedPatch.cols, flippedPatch.rows)) != mirroredContent) {
        return flippedPatch;
    }

    Mat adjustedTextPatch = adaptTextPatchToTargetFill(info, targetFillColor);
    if (adjustedTextPatch.empty()) {
        return flippedPatch;
    }

    adjustedTextPatch.copyTo(flippedPatch(mirroredContent), info.contentMask);
    return flippedPatch;
}

void tintMask(Mat& img, const Mat& mask, const Scalar& color, double alpha) {
    if (img.empty() || mask.empty()) return;
    Mat tint(img.size(), img.type(), color);
    Mat blended;
    addWeighted(img, 1.0 - alpha, tint, alpha, 0.0, blended);
    blended.copyTo(img, mask);
}

void saveBubbleDebugArtifacts(const string& outputPath,
                              const vector<BubbleInfo>& bubbleInfos,
                              const Scalar& targetFillColor) {
    if (!DRAW_DEBUG || outputPath.empty()) return;

    fs::path outPath(outputPath);
    fs::path debugDir = outPath.parent_path() / (outPath.stem().string() + "_debug");
    fs::create_directories(debugDir);
    cout << "Debug dir: " << debugDir.string() << endl;

    for (size_t i = 0; i < bubbleInfos.size(); ++i) {
        const auto& info = bubbleInfos[i];
        string prefix = "bubble_" + to_string(i + 1) + "_";

        auto writeDebug = [&](const string& name, const Mat& img) {
            if (img.empty()) return;
            fs::path file = debugDir / (prefix + name);
            if (!imwrite(file.string(), img)) {
                cerr << "Failed to write debug image: " << file << endl;
            }
        };

        writeDebug("patch.png", info.patch);
        writeDebug("bubble_mask.png", info.visibleMask);
        writeDebug("body_mask.png", info.bodyMaskFull);
        writeDebug("search_mask.png", info.searchMaskFull);
        writeDebug("loose_mask.png", info.looseMaskFull);
        writeDebug("core_mask.png", info.coreMaskFull);
        writeDebug("paste_mask.png", info.pasteMaskFull);
        writeDebug("erase_mask.png", info.eraseMaskFull);

        Mat overlay = info.patch.clone();
        tintMask(overlay, info.bodyMaskFull, Scalar(255, 0, 0), 0.15);
        tintMask(overlay, info.searchMaskFull, Scalar(0, 255, 255), 0.22);
        tintMask(overlay, info.eraseMaskFull, Scalar(255, 0, 255), 0.30);
        tintMask(overlay, info.pasteMaskFull, Scalar(0, 0, 255), 0.35);

        if (info.bodyRectInPatch.area() > 0) {
            rectangle(overlay, info.bodyRectInPatch, Scalar(255, 0, 0), 2, LINE_AA);
        }
        if (info.contentRectInPatch.area() > 0) {
            rectangle(overlay, info.contentRectInPatch, Scalar(0, 0, 255), 2, LINE_AA);
        }
        writeDebug("overlay.png", overlay);

        Mat mirrored = buildMirroredBubblePatch(info, KEEP_BUBBLE_TEXT_READABLE, targetFillColor);
        writeDebug("mirrored_patch.png", mirrored);
    }
}

// ========================
// 通用粘贴与擦除
// ========================

void pastePatchWithMaskSafe(Mat& dst, const Mat& patch, const Mat& mask, const Rect& targetRect) {
    Rect bounds(0, 0, dst.cols, dst.rows);
    Rect clippedTarget = targetRect & bounds;
    if (clippedTarget.width <= 0 || clippedTarget.height <= 0) return;

    int sx = clippedTarget.x - targetRect.x;
    int sy = clippedTarget.y - targetRect.y;
    Rect srcRect(sx, sy, clippedTarget.width, clippedTarget.height);

    patch(srcRect).copyTo(dst(clippedTarget), mask(srcRect));
}

Rect findReferenceGrayPatchRect(const Mat& img) {
    if (img.empty()) return Rect();

    int patchW = min(max(72, img.cols / 9), max(72, img.cols / 4));
    int patchH = min(max(72, img.rows / 10), max(72, img.rows / 5));

    int xStart = min(max(0, static_cast<int>(img.cols * 0.56)), max(0, img.cols - patchW));
    int xEnd = max(xStart, img.cols - patchW - max(12, img.cols / 18));
    int yStart = min(max(0, static_cast<int>(img.rows * 0.15)), max(0, img.rows - patchH));
    int yEnd = max(yStart, min(max(0, img.rows - patchH), static_cast<int>(img.rows * 0.42)));

    int stepX = max(12, patchW / 4);
    int stepY = max(12, patchH / 4);

    Rect bestRect;
    double bestScore = 1e18;

    for (int y = yStart; y <= yEnd; y += stepY) {
        for (int x = xStart; x <= xEnd; x += stepX) {
            Rect r(x, y, patchW, patchH);
            if ((r & Rect(0, 0, img.cols, img.rows)) != r) continue;

            Mat roi = img(r);
            Mat gray, hsv;
            cvtColor(roi, gray, COLOR_BGR2GRAY);
            cvtColor(roi, hsv, COLOR_BGR2HSV);

            Scalar meanGray, stdGray;
            meanStdDev(gray, meanGray, stdGray);

            Mat edges;
            Canny(gray, edges, 30, 90);
            double edgeRatio = static_cast<double>(countNonZero(edges)) / max(1, r.area());

            Scalar meanHsv = mean(hsv);
            double saturation = meanHsv[1];
            double brightnessPenalty = abs(meanGray[0] - 236.0);

            double score = stdGray[0] * 4.0 + edgeRatio * 2500.0 + saturation * 0.8 + brightnessPenalty * 0.4;
            if (score < bestScore) {
                bestScore = score;
                bestRect = r;
            }
        }
    }

    if (bestRect.area() > 0) return bestRect;

    int fallbackX = max(0, img.cols - patchW - max(12, img.cols / 18));
    int fallbackY = min(max(0, static_cast<int>(img.rows * 0.18)), max(0, img.rows - patchH));
    return Rect(fallbackX, fallbackY, patchW, patchH) & Rect(0, 0, img.cols, img.rows);
}

void coverRectWithReferencePatch(Mat& dst, const Mat& referencePatch, const Rect& targetRect) {
    Rect r = targetRect & Rect(0, 0, dst.cols, dst.rows);
    if (r.area() <= 0 || referencePatch.empty()) return;

    Mat resized;
    resize(referencePatch, resized, r.size(), 0, 0, INTER_CUBIC);
    GaussianBlur(resized, resized, Size(15, 15), 0);
    resized.copyTo(dst(r));
}

void shiftImageContentDown(Mat& img, int cutoffY, int shiftY, const Mat& referencePatch) {
    if (img.empty() || shiftY <= 0) return;
    if (cutoffY < 0 || cutoffY >= img.rows) return;

    shiftY = min(shiftY, img.rows - cutoffY);
    if (shiftY <= 0) return;

    Mat original = img.clone();
    int movableHeight = img.rows - cutoffY - shiftY;
    if (movableHeight > 0) {
        original(Rect(0, cutoffY, img.cols, movableHeight))
            .copyTo(img(Rect(0, cutoffY + shiftY, img.cols, movableHeight)));
    }

    coverRectWithReferencePatch(img, referencePatch, Rect(0, cutoffY, img.cols, shiftY));
}

Scalar estimateLocalBackgroundColor(const Mat& img,
                                    const Rect& targetRect,
                                    const Mat& avoidMask) {
    if (img.empty() || targetRect.area() <= 0) return Scalar(240, 240, 240);

    int pad = max(18, max(targetRect.width, targetRect.height) / 3);
    Rect sampleRect = expandRect(targetRect, pad, img.size());
    if (sampleRect.area() <= 0) return Scalar(240, 240, 240);

    Mat sampleMask = Mat::zeros(sampleRect.size(), CV_8UC1);
    rectangle(sampleMask, Rect(0, 0, sampleRect.width, sampleRect.height), Scalar(255), FILLED);

    Rect innerRect(targetRect.x - sampleRect.x,
                   targetRect.y - sampleRect.y,
                   targetRect.width,
                   targetRect.height);
    if (innerRect.area() > 0) {
        rectangle(sampleMask, innerRect, Scalar(0), FILLED);
    }

    if (!avoidMask.empty()) {
        Mat avoidRoi = avoidMask(sampleRect);
        Mat invAvoid;
        bitwise_not(avoidRoi, invAvoid);
        bitwise_and(sampleMask, invAvoid, sampleMask);
    }

    if (countNonZero(sampleMask) < max(80, targetRect.area() / 8)) {
        sampleMask = Mat::zeros(sampleRect.size(), CV_8UC1);

        int topH = max(6, targetRect.height / 3);
        int sideW = max(8, targetRect.width / 4);

        Rect topBand(max(0, innerRect.x - sideW / 2),
                     max(0, innerRect.y - topH - 2),
                     min(sampleRect.width - max(0, innerRect.x - sideW / 2), innerRect.width + sideW),
                     min(topH, innerRect.y));
        Rect bottomBand(max(0, innerRect.x - sideW / 2),
                        min(sampleRect.height, innerRect.y + innerRect.height + 2),
                        min(sampleRect.width - max(0, innerRect.x - sideW / 2), innerRect.width + sideW),
                        max(0, sampleRect.height - min(sampleRect.height, innerRect.y + innerRect.height + 2)));
        Rect leftBand(max(0, innerRect.x - sideW - 2),
                      max(0, innerRect.y),
                      min(sideW, innerRect.x),
                      min(innerRect.height, sampleRect.height - innerRect.y));

        if (topBand.area() > 0) rectangle(sampleMask, topBand, Scalar(255), FILLED);
        if (bottomBand.area() > 0) rectangle(sampleMask, bottomBand, Scalar(255), FILLED);
        if (leftBand.area() > 0) rectangle(sampleMask, leftBand, Scalar(255), FILLED);

        if (!avoidMask.empty()) {
            Mat avoidRoi = avoidMask(sampleRect);
            Mat invAvoid;
            bitwise_not(avoidRoi, invAvoid);
            bitwise_and(sampleMask, invAvoid, sampleMask);
        }
    }

    if (countNonZero(sampleMask) == 0) return Scalar(240, 240, 240);
    return mean(img(sampleRect), sampleMask);
}

void softenRemovedBubbleRegion(Mat& dst,
                               const Mat& input,
                               const BubbleInfo& info,
                               const Mat& avoidMask) {
    PlacedMask placed = buildPlacedBubbleCleanupMask(info, dst.size());
    Rect r = placed.rect & Rect(0, 0, dst.cols, dst.rows);
    if (r.area() <= 0 || placed.mask.empty()) return;

    int blurSize = max(11, ((min(r.width, r.height) / 3) | 1));
    Mat alphaMask;
    GaussianBlur(placed.mask, alphaMask, Size(blurSize, blurSize), 0);

    Mat smoothRoi = dst(r).clone();
    int smoothSize = max(21, ((max(r.width, r.height) / 2) | 1));
    GaussianBlur(smoothRoi, smoothRoi, Size(smoothSize, smoothSize), 0);

    Scalar bgColor = estimateLocalBackgroundColor(input, r, avoidMask);
    Mat fallbackFill(r.size(), dst.type(), bgColor);
    addWeighted(smoothRoi, 0.75, fallbackFill, 0.25, 0.0, smoothRoi);

    Mat dstRoi = dst(r);
    for (int y = 0; y < dstRoi.rows; ++y) {
        const uchar* aRow = alphaMask.ptr<uchar>(y);
        Vec3b* dstRow = dstRoi.ptr<Vec3b>(y);
        const Vec3b* smoothRow = smoothRoi.ptr<Vec3b>(y);

        for (int x = 0; x < dstRoi.cols; ++x) {
            float alpha = aRow[x] / 255.0f;
            if (alpha <= 0.0f) continue;

            for (int c = 0; c < 3; ++c) {
                float blended = (1.0f - alpha) * dstRow[x][c] + alpha * static_cast<float>(smoothRow[x][c]);
                dstRow[x][c] = saturate_cast<uchar>(blended);
            }
        }
    }
}

Mat removeOriginalObjectsByInpaint(const Mat& input,
                                   const vector<AvatarInfo>& avatarInfos,
                                   const vector<BubbleInfo>& bubbleInfos) {
    if (input.empty()) return Mat();

    Mat cleaned = input.clone();

    Rect refRect = findReferenceGrayPatchRect(input);
    Mat referencePatch;
    if (refRect.area() > 0) {
        referencePatch = input(refRect).clone();
    }
    if (referencePatch.empty()) {
        referencePatch = Mat(96, 96, input.type(), Scalar(238, 238, 238));
    }

    for (const auto& info : bubbleInfos) {
        Rect cover = expandRectAsym(info.outerRect, 10, 8, 22, 8, input.size());
        coverRectWithReferencePatch(cleaned, referencePatch, cover);
    }

    for (const auto& info : avatarInfos) {
        Rect cover = expandRectAsym(info.outerRect, 8, 8, 12, 8, input.size());
        coverRectWithReferencePatch(cleaned, referencePatch, cover);
    }

    return cleaned;
}

void drawCenterLine(Mat& img) {
    int centerX = img.cols / 2;

    int outerThickness = max(4, img.cols / 300);
    int innerThickness = max(2, outerThickness / 2);

    line(img, Point(centerX, 0), Point(centerX, img.rows - 1),
         Scalar(0, 0, 0), outerThickness, LINE_AA);
    line(img, Point(centerX, 0), Point(centerX, img.rows - 1),
         Scalar(0, 255, 255), innerThickness, LINE_AA);

    circle(img, Point(centerX, 30), 8, Scalar(0, 0, 255), FILLED, LINE_AA);

    putText(img, "CENTER", Point(max(10, centerX - 45), 65),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 3, LINE_AA);
    putText(img, "CENTER", Point(max(10, centerX - 45), 65),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 1, LINE_AA);
}

Mat processImage(const Mat& input, const string& outputPath, const ProcessOptions& options) {
    if (input.empty()) return Mat();

    vector<Rect> bubbles = findRightGreenBubbles(input);
    vector<Rect> avatarRects = findAllAvatars(input, bubbles);
    Scalar mirroredBubbleFillColor = estimateReferenceWhiteBubbleColor(input);

    vector<BubbleInfo> bubbleInfos = buildBubbleInfos(input, bubbles);
    vector<AvatarInfo> avatarInfos = buildAvatarInfos(input, avatarRects);
    NicknameReferenceStyle nicknameStyle;
    NicknameTextRenderer nicknameRenderer;
    if (options.withNickname) {
        nicknameStyle = estimateNicknameReferenceStyle(input, avatarInfos, bubbleInfos);
        nicknameRenderer = createNicknameTextRenderer(nicknameStyle);
        calibrateNicknameStyleHeights(nicknameStyle, nicknameRenderer);
    }
    saveBubbleDebugArtifacts(outputPath, bubbleInfos, mirroredBubbleFillColor);

    int centerX = input.cols / 2;
    cout << "Mirrored bubble fill=("
         << static_cast<int>(mirroredBubbleFillColor[0]) << ","
         << static_cast<int>(mirroredBubbleFillColor[1]) << ","
         << static_cast<int>(mirroredBubbleFillColor[2]) << ")" << endl;
    cout << "Center X = " << centerX << endl;
    cout << "Bubbles: " << bubbleInfos.size() << endl;
    cout << "Avatars: " << avatarInfos.size() << endl;
    if (options.withNickname) {
        int latinInkHeight = measureNicknameInkHeight("en", true, nicknameStyle.latinFontHeight, nicknameRenderer);
        cout << "Nickname mode: yes (" << options.nickname << ")" << endl;
        cout << "Nickname style: fontHeight=" << nicknameStyle.fontHeight
             << " latinFontHeight=" << nicknameStyle.latinFontHeight
             << " latinInkHeight=" << latinInkHeight
             << " color=(" << static_cast<int>(nicknameStyle.color[0]) << ","
             << static_cast<int>(nicknameStyle.color[1]) << ","
             << static_cast<int>(nicknameStyle.color[2]) << ")"
             << " bubbleYOffset=" << nicknameStyle.bubbleYOffsetFromAvatarTop
             << " bubbleXGap=" << nicknameStyle.bubbleXOffsetFromAvatarRight
             << " refBubbleX=" << nicknameStyle.referenceBubbleX
             << " refNicknameX=" << nicknameStyle.referenceNicknameX
             << " latinFont=" << (nicknameStyle.latinFontPath.empty() ? "<fallback>" : nicknameStyle.latinFontPath)
             << "#" << nicknameStyle.latinFontIndex
             << " cjkFont=" << (nicknameStyle.cjkFontPath.empty() ? "<fallback>" : nicknameStyle.cjkFontPath)
             << "#" << nicknameStyle.cjkFontIndex
             << endl;
    } else {
        cout << "Nickname mode: no" << endl;
    }

    for (size_t i = 0; i < bubbleInfos.size(); ++i) {
        cout << "Bubble " << i + 1
             << " outer=(" << bubbleInfos[i].outerRect.x << "," << bubbleInfos[i].outerRect.y
             << "," << bubbleInfos[i].outerRect.width << "," << bubbleInfos[i].outerRect.height << ")"
             << endl;
    }

    for (size_t i = 0; i < avatarInfos.size(); ++i) {
        Rect cg = contentRectInImage(avatarInfos[i]);
        cout << "Avatar " << i + 1
             << " outer=(" << avatarInfos[i].outerRect.x << "," << avatarInfos[i].outerRect.y
             << "," << avatarInfos[i].outerRect.width << "," << avatarInfos[i].outerRect.height << ")"
             << " content=(" << cg.x << "," << cg.y
             << "," << cg.width << "," << cg.height << ")"
             << endl;
    }

    // 先把原始头像和原始绿色聊天框都擦掉
    Mat output = removeOriginalObjectsByInpaint(input, avatarInfos, bubbleInfos);
    Rect grayRefRect = findReferenceGrayPatchRect(input);
    Mat grayReferencePatch = grayRefRect.area() > 0
        ? input(grayRefRect).clone()
        : Mat(96, 96, input.type(), Scalar(238, 238, 238));

    vector<Rect> targetBubbleRects(bubbleInfos.size());
    vector<char> hasTargetBubble(bubbleInfos.size(), 0);
    vector<Rect> targetAvatarRects(avatarInfos.size());
    vector<char> hasTargetAvatar(avatarInfos.size(), 0);
    vector<Rect> targetNicknameRects(min(bubbleInfos.size(), avatarInfos.size()));
    vector<char> hasTargetNickname(targetNicknameRects.size(), 0);
    Rect nicknameInkBounds;
    Rect nicknameLowercaseBounds;
    Rect nicknameLowercaseSampleBounds;
    bool nicknameHasLowercase = false;
    if (options.withNickname && !options.nickname.empty()) {
        nicknameHasLowercase = hasAsciiLowercase(options.nickname);
        nicknameInkBounds = measureNicknameInkBounds(options.nickname, nicknameStyle, nicknameRenderer, false, 72);
        nicknameLowercaseBounds = measureNicknameInkBounds(options.nickname, nicknameStyle, nicknameRenderer, true, 112);
        nicknameLowercaseSampleBounds = measureNicknameInkBounds("xneo", nicknameStyle, nicknameRenderer, false, 112);
    }

    if (options.withNickname && !options.nickname.empty()) {
        size_t pairCount = min(bubbleInfos.size(), avatarInfos.size());
        int previousBottom = -1000000;

        for (size_t i = 0; i < pairCount; ++i) {
            Rect targetAvatar = mirroredOuterRectByContent(avatarInfos[i], centerX);
            Rect targetAvatarContent = contentRectInTargetImage(avatarInfos[i], targetAvatar);
            Rect targetBubble = mirroredBubbleRect(bubbleInfos[i], centerX);
            Mat flippedBubbleMask;
            flip(bubbleInfos[i].visibleMask, flippedBubbleMask, 1);
            int flippedBubbleTipX = leftmostMaskXInBand(flippedBubbleMask);
            int anchorRight = targetAvatarContent.x + targetAvatarContent.width;
            targetBubble.x = nicknameStyle.referenceBubbleX >= 0
                ? nicknameStyle.referenceBubbleX - max(0, flippedBubbleTipX)
                : anchorRight + nicknameStyle.bubbleXOffsetFromAvatarRight - max(0, flippedBubbleTipX);

            int baseline = 0;
            Size nicknameSize = measureNicknameText(options.nickname, nicknameStyle, nicknameRenderer, &baseline);
            int nicknameVisualLeft = nicknameInkBounds.area() > 0 ? nicknameInkBounds.x : 0;
            int nicknameAlignTop = nicknameHasLowercase
                ? (nicknameLowercaseBounds.area() > 0
                    ? nicknameLowercaseBounds.y
                    : (nicknameLowercaseSampleBounds.area() > 0
                        ? nicknameLowercaseSampleBounds.y
                        : max(1, static_cast<int>(std::round(nicknameStyle.latinFontHeight * 0.58)))))
                : (nicknameInkBounds.area() > 0
                    ? nicknameInkBounds.y
                    : max(1, static_cast<int>(std::round(nicknameStyle.fontHeight * 0.38))));
            int nicknameVisualBottom = nicknameInkBounds.area() > 0
                ? (nicknameInkBounds.y + nicknameInkBounds.height)
                : static_cast<int>(std::round(nicknameStyle.fontHeight * 2.0));
            int nicknameTopY = targetAvatarContent.y - nicknameAlignTop;
            Rect nicknameRect(
                nicknameStyle.referenceNicknameX >= 0
                    ? nicknameStyle.referenceNicknameX - nicknameVisualLeft
                    : anchorRight + nicknameStyle.textXOffsetFromAvatarRight - nicknameVisualLeft,
                nicknameTopY,
                nicknameSize.width,
                max(1, nicknameSize.height)
            );

            int nicknameInkBottom = nicknameRect.y + nicknameVisualBottom;
            targetBubble.y = max(
                targetAvatarContent.y + nicknameStyle.bubbleYOffsetFromAvatarTop,
                nicknameInkBottom + nicknameStyle.nicknameToBubbleGap
            );

            int blockTop = min(targetAvatar.y, nicknameRect.y);
            if (blockTop < previousBottom + nicknameStyle.blockGapY) {
                int shiftY = previousBottom + nicknameStyle.blockGapY - blockTop;
                targetAvatar.y += shiftY;
                targetBubble.y += shiftY;
                nicknameRect.y += shiftY;
                nicknameInkBottom += shiftY;
            }

            previousBottom = max(targetAvatar.y + targetAvatar.height,
                                 max(targetBubble.y + targetBubble.height,
                                     nicknameInkBottom));

            targetAvatarRects[i] = targetAvatar;
            targetBubbleRects[i] = targetBubble;
            targetNicknameRects[i] = nicknameRect;
            hasTargetAvatar[i] = 1;
            hasTargetBubble[i] = 1;
            hasTargetNickname[i] = 1;
        }

        if (pairCount > 0) {
            size_t lastIdx = pairCount - 1;
            Rect oldAvatarRect = mirroredOuterRectByContent(avatarInfos[lastIdx], centerX);
            Rect oldBubbleRect = mirroredBubbleRect(bubbleInfos[lastIdx], centerX);
            int oldBlockBottom = max(oldAvatarRect.y + oldAvatarRect.height,
                                     oldBubbleRect.y + oldBubbleRect.height);
            int newBlockBottom = max(targetAvatarRects[lastIdx].y + targetAvatarRects[lastIdx].height,
                                     targetBubbleRects[lastIdx].y + targetBubbleRects[lastIdx].height);
            int extraShift = max(0, newBlockBottom - oldBlockBottom + nicknameStyle.blockGapY);
            if (extraShift > 0) {
                shiftImageContentDown(output, oldBlockBottom, extraShift, grayReferencePatch);
            }
        }
    }

    // 先贴镜像后的聊天框
    for (size_t i = 0; i < bubbleInfos.size(); ++i) {
        Rect targetBubble = hasTargetBubble[i] ? targetBubbleRects[i] : mirroredBubbleRect(bubbleInfos[i], centerX);
        Mat mirroredPatch = buildMirroredBubblePatch(
            bubbleInfos[i], KEEP_BUBBLE_TEXT_READABLE, mirroredBubbleFillColor
        );
        Mat flippedMask;
        flip(bubbleInfos[i].visibleMask, flippedMask, 1);
        pastePatchWithMaskSafe(output, mirroredPatch, flippedMask, targetBubble);

        if (DRAW_DEBUG && DRAW_BUBBLE_BOX) {
            Rect drawRect = targetBubble & Rect(0, 0, output.cols, output.rows);
            if (drawRect.width > 0 && drawRect.height > 0) {
                rectangle(output, drawRect, Scalar(255, 0, 0), 2, LINE_AA);
                string label = "Bubble " + to_string(i + 1);
                putText(output, label,
                        Point(drawRect.x, max(30, drawRect.y - 10)),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, LINE_AA);
            }
        }
    }

    // 再贴头像
    for (size_t i = 0; i < avatarInfos.size(); ++i) {
        Rect targetOuter = hasTargetAvatar[i] ? targetAvatarRects[i] : mirroredOuterRectByContent(avatarInfos[i], centerX);
        pastePatchWithMaskSafe(output, avatarInfos[i].patch, avatarInfos[i].visibleMask, targetOuter);

        if (DRAW_DEBUG) {
            Rect drawOuter = targetOuter & Rect(0, 0, output.cols, output.rows);
            if (drawOuter.width > 0 && drawOuter.height > 0) {
                rectangle(output, drawOuter, Scalar(0, 0, 255), 3, LINE_AA);
                string label = "Avatar " + to_string(i + 1);
                putText(output, label,
                        Point(drawOuter.x, max(30, drawOuter.y - 10)),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
            }
        }

        if (DRAW_CONTENT_BOX) {
            Rect targetContent(
                targetOuter.x + avatarInfos[i].contentRectInPatch.x,
                targetOuter.y + avatarInfos[i].contentRectInPatch.y,
                avatarInfos[i].contentRectInPatch.width,
                avatarInfos[i].contentRectInPatch.height
            );
            targetContent &= Rect(0, 0, output.cols, output.rows);
            if (targetContent.width > 0 && targetContent.height > 0) {
                rectangle(output, targetContent, Scalar(0, 255, 255), 2, LINE_AA);
            }
        }
    }

    if (options.withNickname && !options.nickname.empty()) {
        for (size_t i = 0; i < targetNicknameRects.size(); ++i) {
            if (!hasTargetNickname[i]) continue;
            drawNicknameText(output, options.nickname, targetNicknameRects[i], nicknameStyle, nicknameRenderer);
        }
    }

    if (DRAW_DEBUG) {
        drawCenterLine(output);
    }
    return output;
}

int main(int argc, char* argv[]) {
    ProcessOptions options;
    string inputPath;
    string outputPath;

    if (argc == 3) {
        inputPath = argv[1];
        outputPath = argv[2];
    } else if (argc == 4 && string(argv[1]) == "nonickname") {
        inputPath = argv[2];
        outputPath = argv[3];
    } else if (argc == 5 && string(argv[1]) == "yesnickname") {
        options.withNickname = true;
        options.nickname = argv[2];
        inputPath = argv[3];
        outputPath = argv[4];
    } else {
        cerr << "Usage: " << argv[0] << " <input> <output>" << endl;
        cerr << "   or: " << argv[0] << " nonickname <input> <output>" << endl;
        cerr << "   or: " << argv[0] << " yesnickname <nickname> <input> <output>" << endl;
        return 1;
    }

    Mat inputImage = imread(inputPath, IMREAD_COLOR);
    if (inputImage.empty()) {
        cerr << "Failed to load image: " << inputPath << endl;
        return 1;
    }

    Mat outputImage = processImage(inputImage, outputPath, options);
    if (outputImage.empty()) {
        cerr << "Processing failed" << endl;
        return 1;
    }

    if (!imwrite(outputPath, outputImage)) {
        cerr << "Failed to save image: " << outputPath << endl;
        return 1;
    }

    cout << "Saved: " << outputPath << endl;
    return 0;
}
