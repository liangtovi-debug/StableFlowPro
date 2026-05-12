#include <metal_stdlib>
using namespace metal;

// ============================================================
// MetalStabilizer 需要的 shader 函数
// ============================================================

// 分析两帧之间的运动（全局光流）
kernel void analyzeMotion(
    texture2d<float, access::read> prevTex [[texture(0)]],
    texture2d<float, access::read> currTex [[texture(1)]],
    device atomic_uint *sumX [[buffer(0)]],
    device atomic_uint *sumY [[buffer(1)]],
    device atomic_uint *totalWeight [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = prevTex.get_width();
    int h = prevTex.get_height();
    
    // 边界检查
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }
    
    // 每隔一定像素采样（加速）
    if ((gid.x + gid.y) % 4 != 0) {
        return;
    }
    
    // 简单的块匹配
    int blockSize = 8;
    int bestDx = 0;
    int bestDy = 0;
    float bestError = 1e9;
    
    for (int dy = -4; dy <= 4; dy++) {
        for (int dx = -4; dx <= 4; dx++) {
            float error = 0;
            int count = 0;
            
            for (int by = -blockSize/2; by < blockSize/2; by++) {
                for (int bx = -blockSize/2; bx < blockSize/2; bx++) {
                    int2 pos = int2(gid) + int2(dx + bx, dy + by);
                    if (pos.x >= 0 && pos.x < w && pos.y >= 0 && pos.y < h) {
                        float p1 = prevTex.read(uint2(pos)).r;
                        float p2 = currTex.read(uint2(pos.x - dx, pos.y - dy)).r;
                        error += fabs(p1 - p2);
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                error /= count;
                if (error < bestError) {
                    bestError = error;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
    }
    
    // 只有误差小于阈值才接受
    if (bestError < 0.1) {
        atomic_fetch_add_explicit(sumX, uint(bestDx * 1000), memory_order_relaxed);
        atomic_fetch_add_explicit(sumY, uint(bestDy * 1000), memory_order_relaxed);
        atomic_fetch_add_explicit(totalWeight, 1, memory_order_relaxed);
    }
}

// 稳定帧（带crop的变换）
kernel void stabilizeFrame(
    texture2d<float, access::sample> inputTex [[texture(0)]],
    texture2d<float, access::write> outputTex [[texture(1)]],
    constant float &offsetX [[buffer(0)]],
    constant float &offsetY [[buffer(1)]],
    constant float &cropX [[buffer(2)]],
    constant float &cropY [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = outputTex.get_width();
    int h = outputTex.get_height();
    
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }
    
    // 计算输出像素对应的输入坐标（带偏移和crop）
    float u = (float(gid.x) + cropX) / float(w) + offsetX;
    float v = (float(gid.y) + cropY) / float(h) + offsetY;
    
    // 边界检查
    if (u < 0 || u > 1 || v < 0 || v > 1) {
        outputTex.write(float4(0, 0, 0, 1), gid);
        return;
    }
    
    // 采样
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float4 color = inputTex.sample(s, float2(u, v));
    
    outputTex.write(color, gid);
}

// ============================================================
// 梯度光流：使用 Sobel 梯度而非亮度差
kernel void opticalFlowGradient(
    texture2d<float, access::read> prevTex [[texture(0)]],
    texture2d<float, access::read> currTex [[texture(1)]],
    device atomic_uint *resultX [[buffer(0)]],
    device atomic_uint *resultY [[buffer(1)]],
    device atomic_uint *resultCount [[buffer(2)]],
    constant float2 &point [[buffer(3)]],
    constant float2 &size [[buffer(4)]],
    constant int &radius [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int winSize = 3;
    int r = radius;
    
    // 归一化坐标转像素坐标
    float2 pt = point * size;
    int px = int(pt.x);
    int py = int(pt.y);
    
    // 边界检查
    if (px < r + winSize || px >= int(size.x) - r - winSize ||
        py < r + winSize || py >= int(size.y) - r - winSize) {
        return;
    }
    
    float bestError = 1e9;
    int bestDx = 0;
    int bestDy = 0;
    
    // 预计算前一帧的梯度
    float gx[5][5];
    float gy[5][5];
    
    for (int wy = -winSize; wy <= winSize; wy++) {
        for (int wx = -winSize; wx <= winSize; wx++) {
            // Sobel X: [-1, 0, 1]
            float left = prevTex.read(uint2(px + wx - 1, py + wy)).r;
            float right = prevTex.read(uint2(px + wx + 1, py + wy)).r;
            gx[wx+winSize][wy+winSize] = fabs(right - left);
            
            // Sobel Y: [-1, 0, 1]^T
            float up = prevTex.read(uint2(px + wx, py + wy - 1)).r;
            float down = prevTex.read(uint2(px + wx, py + wy + 1)).r;
            gy[wx+winSize][wy+winSize] = fabs(down - up);
        }
    }
    
    // 搜索最佳匹配
    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            float error = 0;
            float count = 0;
            
            for (int wy = -winSize; wy <= winSize; wy++) {
                for (int wx = -winSize; wx <= winSize; wx++) {
                    int wx1 = wx + winSize;
                    int wy1 = wy + winSize;
                    
                    // 前一帧梯度
                    float gx1 = gx[wx1][wy1];
                    float gy1 = gy[wx1][wy1];
                    
                    // 当前帧梯度
                    float left2 = currTex.read(uint2(px + wx + dx - 1, py + wy + dy)).r;
                    float right2 = currTex.read(uint2(px + wx + dx + 1, py + wy + dy)).r;
                    float up2 = currTex.read(uint2(px + wx + dx, py + wy + dy - 1)).r;
                    float down2 = currTex.read(uint2(px + wx + dx, py + wy + dy + 1)).r;
                    
                    float gx2 = fabs(right2 - left2);
                    float gy2 = fabs(down2 - up2);
                    
                    // 梯度差 (比亮度差更稳定)
                    error += fabs(gx1 - gx2) + fabs(gy1 - gy2);
                    count += 1.0;
                }
            }
            
            error /= count;
            
            if (error < bestError) {
                bestError = error;
                bestDx = dx;
                bestDy = dy;
            }
        }
    }
    
    // 只有误差在阈值内才接受
    float confidenceThreshold = 0.15;
    if (bestError < confidenceThreshold) {
        atomic_fetch_add_explicit(resultX, bestDx, memory_order_relaxed);
        atomic_fetch_add_explicit(resultY, bestDy, memory_order_relaxed);
        atomic_fetch_add_explicit(resultCount, 1, memory_order_relaxed);
    }
}

// 稳定渲染 shader
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut stableVertex(uint vertexID [[vertex_id]]) {
    float2 positions[4] = {
        {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}
    };
    float2 texCoords[4] = {
        {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}
    };
    
    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 stableFragment(
    VertexOut in [[stage_in]],
    texture2d<float> inputTexture [[texture(0)]],
    constant float2 &offset [[buffer(0)]],
    constant float &scale [[buffer(1)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    
    float2 center = float2(0.5, 0.5);
    float2 uv = (in.texCoord - center) / scale + center;
    uv += offset;
    
    return inputTexture.sample(s, uv);
}

// ============================================================
// 模板匹配 Kernel (全局搜索)
// 用于"重定位" - 当光流失败时找回目标
// ============================================================
kernel void templateMatch(
    texture2d<float, access::read> currTex [[texture(0)]],
    texture2d<float, access::read> templateTex [[texture(1)]],
    device float2 *bestPos [[buffer(0)]],
    device float *bestScore [[buffer(1)]],
    constant float2 &center [[buffer(2)]],
    constant float2 &imageSize [[buffer(3)]],
    constant float &searchRadius [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 每个线程负责搜索范围内的一个候选位置
    int searchR = int(searchRadius);
    int templateW = templateTex.get_width();
    int templateH = templateTex.get_height();
    
    // 当前线程对应的搜索偏移
    int totalWidth = searchR * 2 + 1;
    int dx = int(gid.x) - searchR;
    int dy = int(gid.y) - searchR;
    
    // 越界检查
    if (dx < -searchR || dx > searchR || dy < -searchR || dy > searchR) {
        return;
    }
    
    // 计算候选位置（归一化坐标）
    float2 candidatePos = float2(center.x + float(dx) / imageSize.x, center.y + float(dy) / imageSize.y);
    
    // 边界检查 - 确保模板完整
    int px = int(candidatePos.x * imageSize.x);
    int py = int(candidatePos.y * imageSize.y);
    
    int halfW = templateW / 2;
    int halfH = templateH / 2;
    
    if (px - halfW < 0 || px + halfW >= int(imageSize.x) ||
        py - halfH < 0 || py + halfH >= int(imageSize.y)) {
        return;
    }
    
    // 计算 SSD (Sum of Squared Differences)
    float error = 0.0;
    int count = 0;
    
    for (int ty = 0; ty < templateH; ty++) {
        for (int tx = 0; tx < templateW; tx++) {
            float v1 = templateTex.read(uint2(tx, ty)).r;
            
            int sx = px - halfW + tx;
            int sy = py - halfH + ty;
            
            float v2 = currTex.read(uint2(sx, sy)).r;
            
            float diff = v1 - v2;
            error += diff * diff;
            count++;
        }
    }
    
    error /= float(count);
    
    // 每个线程写入自己对应的槽位，CPU侧遍历找最小误差
    int idx = int(gid.x) + int(gid.y) * totalWidth;
    bestPos[idx] = candidatePos;
    bestScore[idx] = error;
}

// 并行模板匹配 - 每个线程计算一个候选位置
kernel void templateMatchParallel(
    texture2d<float, access::read> currTex [[texture(0)]],
    texture2d<float, access::read> templateTex [[texture(1)]],
    device atomic_float *errors [[buffer(0)]],
    constant float2 &center [[buffer(1)]],
    constant float2 &imageSize [[buffer(2)]],
    constant float &searchRadius [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int searchR = int(searchRadius);
    int templateW = templateTex.get_width();
    int templateH = templateTex.get_height();
    
    int totalWidth = searchR * 2 + 1;
    
    // 当前线程对应的搜索偏移
    int idx = int(gid.x) + int(gid.y) * totalWidth;
    int dx = idx % totalWidth - searchR;
    int dy = idx / totalWidth - searchR;
    
    // 越界检查
    if (dx < -searchR || dx > searchR || dy < -searchR || dy > searchR ||
        idx >= totalWidth * totalWidth) {
        return;
    }
    
    // 计算候选位置的像素坐标
    float2 candidatePos = float2(
        center.x + float(dx) / imageSize.x,
        center.y + float(dy) / imageSize.y
    );
    
    int px = int(candidatePos.x * imageSize.x);
    int py = int(candidatePos.y * imageSize.y);
    
    int halfW = templateW / 2;
    int halfH = templateH / 2;
    
    // 边界检查
    if (px - halfW < 0 || px + halfW >= int(imageSize.x) ||
        py - halfH < 0 || py + halfH >= int(imageSize.y)) {
        atomic_store_explicit(&errors[idx], 1e9, memory_order_relaxed);
        return;
    }
    
    // 计算 SSD
    float error = 0.0;
    int count = 0;
    
    for (int ty = 0; ty < templateH; ty++) {
        for (int tx = 0; tx < templateW; tx++) {
            float v1 = templateTex.read(uint2(tx, ty)).r;
            float v2 = currTex.read(uint2(px - halfW + tx, py - halfH + ty)).r;
            float diff = v1 - v2;
            error += diff * diff;
            count++;
        }
    }
    
    error /= float(count);
    
    // 存储误差（误差越小越好）
    atomic_store_explicit(&errors[idx], error, memory_order_relaxed);
}

// 提取模板 patch（CPU侧更简单，用GPU反而麻烦）
kernel void extractPatch(
    texture2d<float, access::read> sourceTex [[texture(0)]],
    texture2d<float, access::write> patchTex [[texture(1)]],
    constant float2 &center [[buffer(0)]],
    constant float2 &imageSize [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int patchW = patchTex.get_width();
    int patchH = patchTex.get_height();
    
    // 边界检查
    if (gid.x >= uint(patchW) || gid.y >= uint(patchH)) {
        return;
    }
    
    // 计算源图像坐标
    int px = int(center.x * imageSize.x) - patchW / 2 + int(gid.x);
    int py = int(center.y * imageSize.y) - patchH / 2 + int(gid.y);
    
    // 边界保护
    px = max(0, min(px, int(imageSize.x) - 1));
    py = max(0, min(py, int(imageSize.y) - 1));
    
    float value = sourceTex.read(uint2(px, py)).r;
    patchTex.write(float4(value, value, value, 1.0), gid);
}

// ============================================================
// 并行多点光流 - 一次dispatch处理9个点
// ============================================================

// 每个线程跟踪一个点，9个线程并行
kernel void opticalFlowMultiPoint(
    texture2d<float, access::read> prevTex [[texture(0)]],
    texture2d<float, access::read> currTex [[texture(1)]],
    device atomic_uint *resultX [[buffer(0)]],
    device atomic_uint *resultY [[buffer(1)]],
    device atomic_uint *resultCount [[buffer(2)]],
    constant float2 *points [[buffer(3)]],        // 9个点（归一化坐标）
    constant float2 &size [[buffer(4)]],          // 图像尺寸
    constant int &radius [[buffer(5)]],           // 搜索半径
    constant int &numPoints [[buffer(6)]],        // 点数量
    uint tid [[thread_position_in_grid]]
) {
    // 每个线程处理一个点
    if (tid >= uint(numPoints)) return;
    
    float2 point = points[tid];
    int winSize = 3;
    int r = radius;
    
    // 归一化坐标转像素坐标
    float2 pt = point * size;
    int px = int(pt.x);
    int py = int(pt.y);
    
    // 边界检查
    if (px < r + winSize || px >= int(size.x) - r - winSize ||
        py < r + winSize || py >= int(size.y) - r - winSize) {
        return;
    }
    
    float bestError = 1e9;
    int bestDx = 0;
    int bestDy = 0;
    
    // 预计算前一帧的梯度
    float gx[5][5];
    float gy[5][5];
    
    for (int wy = -winSize; wy <= winSize; wy++) {
        for (int wx = -winSize; wx <= winSize; wx++) {
            float left = prevTex.read(uint2(px + wx - 1, py + wy)).r;
            float right = prevTex.read(uint2(px + wx + 1, py + wy)).r;
            gx[wx+winSize][wy+winSize] = fabs(right - left);
            
            float up = prevTex.read(uint2(px + wx, py + wy - 1)).r;
            float down = prevTex.read(uint2(px + wx, py + wy + 1)).r;
            gy[wx+winSize][wy+winSize] = fabs(down - up);
        }
    }
    
    // 搜索最佳匹配
    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            float error = 0;
            float count = 0;
            
            for (int wy = -winSize; wy <= winSize; wy++) {
                for (int wx = -winSize; wx <= winSize; wx++) {
                    int wx1 = wx + winSize;
                    int wy1 = wy + winSize;
                    
                    float gx1 = gx[wx1][wy1];
                    float gy1 = gy[wx1][wy1];
                    
                    float left2 = currTex.read(uint2(px + wx + dx - 1, py + wy + dy)).r;
                    float right2 = currTex.read(uint2(px + wx + dx + 1, py + wy + dy)).r;
                    float up2 = currTex.read(uint2(px + wx + dx, py + wy + dy - 1)).r;
                    float down2 = currTex.read(uint2(px + wx + dx, py + wy + dy + 1)).r;
                    
                    float gx2 = fabs(right2 - left2);
                    float gy2 = fabs(down2 - up2);
                    
                    error += fabs(gx1 - gx2) + fabs(gy1 - gy2);
                    count += 1.0;
                }
            }
            
            error /= count;
            
            if (error < bestError) {
                bestError = error;
                bestDx = dx;
                bestDy = dy;
            }
        }
    }
    
    // 只有误差在阈值内才接受
    float confidenceThreshold = 0.15;
    if (bestError < confidenceThreshold) {
        // 使用 tid 作为索引区分不同点
        atomic_fetch_add_explicit(resultX, bestDx, memory_order_relaxed);
        atomic_fetch_add_explicit(resultY, bestDy, memory_order_relaxed);
        atomic_fetch_add_explicit(resultCount, 1, memory_order_relaxed);
    }
}
