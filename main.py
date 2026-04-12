import cv2
import numpy as np
import sys
import os
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont

# --- CONSTANTS & CONFIG ---
DRAW_DEBUG = False
DRAW_CONTENT_BOX = False
DRAW_BUBBLE_BOX = False
KEEP_BUBBLE_TEXT_READABLE = True
DEFAULT_MIRRORED_BUBBLE_FILL = (255, 255, 255)
NICKNAME_FONT_SCALE = 1.5

# --- DATA CLASSES ---
@dataclass
class AvatarContentResult:
    contentRect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    visibleMask: Optional[np.ndarray] = None

@dataclass
class AvatarInfo:
    outerRect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contentRectInPatch: Tuple[int, int, int, int] = (0, 0, 0, 0)
    patch: Optional[np.ndarray] = None
    visibleMask: Optional[np.ndarray] = None

@dataclass
class BubbleInfo:
    outerRect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    patch: Optional[np.ndarray] = None
    visibleMask: Optional[np.ndarray] = None
    fillColor: Tuple[float, float, float] = (0, 0, 0)
    bodyRectInPatch: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contentRectInPatch: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contentPatch: Optional[np.ndarray] = None
    contentMask: Optional[np.ndarray] = None
    pasteMaskFull: Optional[np.ndarray] = None
    eraseMaskFull: Optional[np.ndarray] = None
    bodyMaskFull: Optional[np.ndarray] = None
    searchMaskFull: Optional[np.ndarray] = None
    coreMaskFull: Optional[np.ndarray] = None
    looseMaskFull: Optional[np.ndarray] = None
    hasContent: bool = False

@dataclass
class BubbleContentResult:
    bodyRect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contentRect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    visibleMask: Optional[np.ndarray] = None
    pasteMaskFull: Optional[np.ndarray] = None
    eraseMaskFull: Optional[np.ndarray] = None
    bodyMaskFull: Optional[np.ndarray] = None
    searchMaskFull: Optional[np.ndarray] = None
    coreMaskFull: Optional[np.ndarray] = None
    looseMaskFull: Optional[np.ndarray] = None

@dataclass
class PlacedMask:
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mask: Optional[np.ndarray] = None

@dataclass
class ProcessOptions:
    withNickname: bool = False
    nickname: str = ""

@dataclass
class NicknameReferenceStyle:
    valid: bool = False
    color: Tuple[float, float, float] = (133, 133, 133)
    fontHeight: int = 24
    latinFontHeight: int = 24
    textXOffsetFromAvatarRight: int = 29
    textTopOffsetFromAvatarTop: int = -2
    bubbleXOffsetFromAvatarRight: int = 17
    bubbleYOffsetFromAvatarTop: int = 29
    nicknameToBubbleGap: int = 4
    blockGapY: int = 8
    referenceAvatarSize: int = 95
    latinFontPath: str = ""
    cjkFontPath: str = ""
    latinFontIndex: int = 0
    cjkFontIndex: int = 0
    referenceBubbleX: int = -1
    referenceNicknameX: int = -1

@dataclass
class NicknameTextRenderer:
    latin: Optional[ImageFont.FreeTypeFont] = None
    cjk: Optional[ImageFont.FreeTypeFont] = None

# --- UTILITY MATH/RECT FUNCTIONS ---
def rect_area(r: Tuple[int, int, int, int]) -> int:
    return max(0, r[2]) * max(0, r[3])

def rect_and(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1 = max(r1[0], r2[0]), max(r1[1], r2[1])
    x2, y2 = min(r1[0] + r1[2], r2[0] + r2[2]), min(r1[1] + r1[3], r2[1] + r2[3])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return (0, 0, 0, 0)
    return (int(x1), int(y1), int(w), int(h))

def rect_or(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if rect_area(r1) == 0: return r2
    if rect_area(r2) == 0: return r1
    x1, y1 = min(r1[0], r2[0]), min(r1[1], r2[1])
    x2, y2 = max(r1[0] + r1[2], r2[0] + r2[2]), max(r1[1] + r1[3], r2[1] + r2[3])
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def expandRect(r: Tuple[int, int, int, int], pad: int, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    r_expanded = (r[0] - pad, r[1] - pad, r[2] + 2 * pad, r[3] + 2 * pad)
    return rect_and(r_expanded, (0, 0, size[0], size[1]))

def insetRect(r: Tuple[int, int, int, int], dx: int, dy: int, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    if r[2] <= 2 * dx or r[3] <= 2 * dy:
        return rect_and(r, (0, 0, size[0], size[1]))
    r_inset = (r[0] + dx, r[1] + dy, r[2] - 2 * dx, r[3] - 2 * dy)
    return rect_and(r_inset, (0, 0, size[0], size[1]))

def expandRectAsym(r: Tuple[int, int, int, int], left: int, top: int, right: int, bottom: int, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    r_expanded = (r[0] - left, r[1] - top, r[2] + left + right, r[3] + top + bottom)
    return rect_and(r_expanded, (0, 0, size[0], size[1]))

def bboxFromMaskPeakThreshold(mask: np.ndarray, colPeakRatio: float, rowPeakRatio: float) -> Tuple[int, int, int, int]:
    if mask is None or mask.size == 0 or len(mask.shape) != 2: return (0, 0, 0, 0)
    colCounts = np.count_nonzero(mask, axis=0)
    rowCounts = np.count_nonzero(mask, axis=1)
    maxColCount = np.max(colCounts) if colCounts.size > 0 else 0
    maxRowCount = np.max(rowCounts) if rowCounts.size > 0 else 0
    if maxColCount == 0 or maxRowCount == 0: return (0, 0, 0, 0)
    colThr = max(1, int(maxColCount * colPeakRatio))
    rowThr = max(1, int(maxRowCount * rowPeakRatio))
    col_idx = np.where(colCounts >= colThr)[0]
    row_idx = np.where(rowCounts >= rowThr)[0]
    if len(col_idx) == 0 or len(row_idx) == 0: return (0, 0, 0, 0)
    x1, x2 = col_idx[0], col_idx[-1]
    y1, y2 = row_idx[0], row_idx[-1]
    if x1 < 0 or x2 < x1 or y1 < 0 or y2 < y1: return (0, 0, 0, 0)
    return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))

def boundingRectOfMask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    if mask is None or mask.size == 0: return (0, 0, 0, 0)
    pts = cv2.findNonZero(mask)
    if pts is None: return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(pts)
    return (int(x), int(y), int(w), int(h))

def leftmostMaskXInBand(mask: np.ndarray, topRatio=0.28, bottomRatio=0.72) -> int:
    if mask is None or mask.size == 0: return -1
    y1 = max(0, min(mask.shape[0] - 1, int(math.floor(mask.shape[0] * topRatio))))
    y2 = max(y1 + 1, min(mask.shape[0], int(math.ceil(mask.shape[0] * bottomRatio))))
    band = mask[y1:y2, :]
    colCounts = np.count_nonzero(band, axis=0)
    nz_cols = np.where(colCounts > 0)[0]
    if len(nz_cols) > 0: return int(nz_cols[0])
    bounds = boundingRectOfMask(mask)
    return int(bounds[0]) if rect_area(bounds) > 0 else -1

def hasAsciiLowercase(text: str) -> bool:
    for ch in text:
        if 'a' <= ch <= 'z': return True
    return False

def looksLikeAvatar(roi: np.ndarray) -> bool:
    if roi is None or roi.size == 0: return False
    ratio = roi.shape[1] / max(1, roi.shape[0])
    if ratio < 0.75 or ratio > 1.35: return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(gray)
    if m[0] < 140: return False
    edges = cv2.Canny(gray, 50, 150)
    edgeRatio = cv2.countNonZero(edges) / (roi.shape[0] * roi.shape[1])
    if edgeRatio < 0.02: return False
    return True

def estimateBackgroundFromCorners(patch: np.ndarray) -> Tuple[float, float, float]:
    cw = max(2, patch.shape[1] // 6)
    ch = max(2, patch.shape[0] // 6)
    corners = [
        (0, 0, cw, ch),
        (patch.shape[1] - cw, 0, cw, ch),
        (0, patch.shape[0] - ch, cw, ch),
        (patch.shape[1] - cw, patch.shape[0] - ch, cw, ch)
    ]
    sumC = [0.0, 0.0, 0.0]
    count = 0
    for rect in corners:
        x, y, w, h = rect
        roi = patch[y:y+h, x:x+w]
        m = cv2.mean(roi)
        sumC[0] += m[0]
        sumC[1] += m[1]
        sumC[2] += m[2]
        count += 1
    if count == 0: return (200.0, 200.0, 200.0)
    return (sumC[0] / count, sumC[1] / count, sumC[2] / count)

def bboxFromMaskProjection(mask: np.ndarray, colRatio: float, rowRatio: float) -> Tuple[int, int, int, int]:
    if mask is None or mask.size == 0 or len(mask.shape) != 2: return (0, 0, 0, 0)
    colThr = max(1, int(mask.shape[0] * colRatio))
    rowThr = max(1, int(mask.shape[1] * rowRatio))
    colCounts = np.count_nonzero(mask, axis=0)
    rowCounts = np.count_nonzero(mask, axis=1)
    col_idx = np.where(colCounts >= colThr)[0]
    row_idx = np.where(rowCounts >= rowThr)[0]
    if len(col_idx) == 0 or len(row_idx) == 0: return (0, 0, 0, 0)
    x1, x2 = col_idx[0], col_idx[-1]
    y1, y2 = row_idx[0], row_idx[-1]
    if x1 < 0 or x2 < x1 or y1 < 0 or y2 < y1: return (0, 0, 0, 0)
    return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))

# --- AVATAR & BUBBLE DETECTION ---
import functools

def extractWhiteBubbleMask(img: np.ndarray, bubbleRect: Tuple[int, int, int, int]) -> np.ndarray:
    if img is None or img.size == 0 or rect_area(bubbleRect) <= 0: return np.zeros((1,1), dtype=np.uint8)
    roiRect = expandRect(bubbleRect, 3, (img.shape[1], img.shape[0]))
    roi = img[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    whiteMask = cv2.inRange(hsv, np.array([0, 0, 215]), np.array([180, 45, 255]))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, kernel3)
    
    localBubble = (bubbleRect[0] - roiRect[0], bubbleRect[1] - roiRect[1], bubbleRect[2], bubbleRect[3])
    localBubble = rect_and(localBubble, (0, 0, roi.shape[1], roi.shape[0]))
    if rect_area(localBubble) <= 0: return whiteMask
    
    contours, _ = cv2.findContours(whiteMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bestScore = -1.0
    bestIdx = -1
    localCenter = (localBubble[0] + localBubble[2] // 2, localBubble[1] + localBubble[3] // 2)
    
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        overlap_rect = rect_and((x, y, w, h), localBubble)
        overlap = rect_area(overlap_rect)
        containsCenter = cv2.pointPolygonTest(c, localCenter, False) >= 0
        score = overlap + (rect_area(localBubble) if containsCenter else 0.0)
        if score > bestScore:
            bestScore = score
            bestIdx = i
            
    if bestIdx < 0: return whiteMask
    chosen = np.zeros(whiteMask.shape, dtype=np.uint8)
    cv2.drawContours(chosen, contours, bestIdx, 255, cv2.FILLED, cv2.LINE_8)
    return chosen

def findRightGreenBubbles(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 60, 120]), np.array([70, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 2500: continue
        if w < 50 or h < 35: continue
        if x + w < img.shape[1] * 0.8: continue
        bubbles.append((x, y, w, h))
        
    def cmp(a, b):
        if abs(a[1] - b[1]) > 10: return -1 if a[1] < b[1] else 1
        return -1 if a[0] < b[0] else (1 if a[0] > b[0] else 0)
    bubbles.sort(key=functools.cmp_to_key(cmp))
    return bubbles

def findLeftWhiteBubbles(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    bubbles = []
    if img is None or img.size == 0: return bubbles
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    whiteMask = cv2.inRange(hsv, np.array([0, 0, 215]), np.array([180, 42, 255]))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_OPEN, kernel3)
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, kernel5)
    contours, _ = cv2.findContours(whiteMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 1100: continue
        if w < 42 or h < 24: continue
        if w > img.shape[1] * 0.75: continue
        if x > img.shape[1] * 0.58: continue
        if y < img.shape[0] * 0.06 or y > img.shape[0] * 0.90: continue
        if y < img.shape[0] * 0.18 and w > img.shape[1] * 0.45: continue
        ratio = float(w) / max(1, h)
        if ratio < 0.8 or ratio > 9.5: continue
        bubbles.append((x, y, w, h))
        
    def cmp(a, b):
        if abs(a[1] - b[1]) > 8: return -1 if a[1] < b[1] else 1
        return -1 if a[0] < b[0] else (1 if a[0] > b[0] else 0)
    bubbles.sort(key=functools.cmp_to_key(cmp))
    return bubbles

def findAllAvatars(img: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    avatars = []
    for b in bubbles:
        searchX = min(b[0] + b[2] + 5, img.shape[1] - 1)
        searchY = max(b[1] - 20, 0)
        searchW = min(180, img.shape[1] - searchX)
        searchH = min(max(b[3], 140) + 40, img.shape[0] - searchY)
        if searchW <= 0 or searchH <= 0: continue
        
        searchRect = (searchX, searchY, searchW, searchH)
        searchROI = img[searchY:searchY+searchH, searchX:searchX+searchW]
        gray = cv2.cvtColor(searchROI, cv2.COLOR_BGR2GRAY)
        _, brightMask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        brightMask = cv2.morphologyEx(brightMask, cv2.MORPH_OPEN, kernel)
        brightMask = cv2.morphologyEx(brightMask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(brightMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 3500: continue
            if w < 60 or h < 60: continue
            globalRect = (searchRect[0] + x, searchRect[1] + y, w, h)
            globalRect = rect_and(globalRect, (0, 0, img.shape[1], img.shape[0]))
            if globalRect[2] <= 0 or globalRect[3] <= 0: continue
            
            duplicate = False
            for a in avatars:
                inter = rect_and(a, globalRect)
                if rect_area(inter) > 0.6 * min(rect_area(a), rect_area(globalRect)):
                    duplicate = True
                    break
            if duplicate: continue
            
            candidate = img[globalRect[1]:globalRect[1]+globalRect[3], globalRect[0]:globalRect[0]+globalRect[2]]
            if looksLikeAvatar(candidate):
                avatars.append(globalRect)
                
    def cmp(a, b):
        if abs(a[1] - b[1]) > 10: return -1 if a[1] < b[1] else 1
        return -1 if a[0] < b[0] else (1 if a[0] > b[0] else 0)
    avatars.sort(key=functools.cmp_to_key(cmp))
    return avatars

def detectVisibleAvatarContent(patch: np.ndarray) -> AvatarContentResult:
    result = AvatarContentResult(contentRect=(0, 0, patch.shape[1], patch.shape[0]), visibleMask=np.zeros(patch.shape[:2], dtype=np.uint8))
    if patch is None or patch.size == 0: return result
    bg = estimateBackgroundFromCorners(patch)
    bgMat = np.full(patch.shape, bg, dtype=patch.dtype)
    diffBgr = cv2.absdiff(patch, bgMat)
    ch = cv2.split(diffBgr)
    maxDiff = np.maximum(np.maximum(ch[0], ch[1]), ch[2])
    _, diffMask = cv2.threshold(maxDiff, 14, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel3)
    combined = cv2.bitwise_or(diffMask, edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel5)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel3)
    
    projRect = bboxFromMaskProjection(combined, 0.06, 0.06)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bestScore = -1e18
    bestIdx = -1
    center = (patch.shape[1] / 2.0, patch.shape[0] / 2.0)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < patch.shape[0] * patch.shape[1] * 0.03: continue
        x, y, w, h = cv2.boundingRect(c)
        rc = (x + w / 2.0, y + h / 2.0)
        dist = math.hypot(rc[0] - center[0], rc[1] - center[1])
        score = area - 0.15 * dist * dist
        if score > bestScore:
            bestScore = score
            bestIdx = i
            
    contourRect = (0,0,0,0)
    contourMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    if bestIdx >= 0:
        cv2.drawContours(contourMask, contours, bestIdx, 255, cv2.FILLED)
        contourMask = cv2.morphologyEx(contourMask, cv2.MORPH_CLOSE, kernel5)
        contourRect = cv2.boundingRect(contours[bestIdx])
        
    finalRect = (0,0,0,0)
    if rect_area(contourRect) > 0 and rect_area(projRect) > 0:
        finalRect = rect_or(contourRect, projRect)
    elif rect_area(contourRect) > 0:
        finalRect = contourRect
    elif rect_area(projRect) > 0:
        finalRect = projRect
        cv2.rectangle(contourMask, (finalRect[0], finalRect[1]), (finalRect[0]+finalRect[2], finalRect[1]+finalRect[3]), 255, cv2.FILLED)
    else:
        finalRect = (0, 0, patch.shape[1], patch.shape[0])
        contourMask[:] = 255
        
    finalRect = expandRect(finalRect, 2, (patch.shape[1], patch.shape[0]))
    if rect_area(finalRect) < patch.shape[0] * patch.shape[1] * 0.15:
        finalRect = (0, 0, patch.shape[1], patch.shape[0])
        contourMask[:] = 255
        
    solidMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.rectangle(solidMask, (finalRect[0], finalRect[1]), (finalRect[0]+finalRect[2], finalRect[1]+finalRect[3]), 255, cv2.FILLED)
    result.contentRect = finalRect
    result.visibleMask = solidMask
    return result

def buildAvatarInfos(img: np.ndarray, avatarRects: List[Tuple[int, int, int, int]]) -> List[AvatarInfo]:
    infos = []
    for r in avatarRects:
        if r[2] <= 0 or r[3] <= 0: continue
        patch = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].copy()
        det = detectVisibleAvatarContent(patch)
        info = AvatarInfo(outerRect=r, contentRectInPatch=det.contentRect, patch=patch, visibleMask=det.visibleMask)
        infos.append(info)
    return infos

def contentRectInImage(info: AvatarInfo) -> Tuple[int, int, int, int]:
    return (info.outerRect[0] + info.contentRectInPatch[0], info.outerRect[1] + info.contentRectInPatch[1],
            info.contentRectInPatch[2], info.contentRectInPatch[3])

def contentRectInTargetImage(info: AvatarInfo, targetOuter: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (targetOuter[0] + info.contentRectInPatch[0], targetOuter[1] + info.contentRectInPatch[1],
            info.contentRectInPatch[2], info.contentRectInPatch[3])

def detectAvatarContentRectInImage(img: np.ndarray, avatarOuterRect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if img is None or img.size == 0 or rect_area(avatarOuterRect) <= 0: return avatarOuterRect
    patch = img[avatarOuterRect[1]:avatarOuterRect[1]+avatarOuterRect[3], avatarOuterRect[0]:avatarOuterRect[0]+avatarOuterRect[2]]
    det = detectVisibleAvatarContent(patch.copy())
    local = det.contentRect if rect_area(det.contentRect) > 0 else (0, 0, avatarOuterRect[2], avatarOuterRect[3])
    return (avatarOuterRect[0] + local[0], avatarOuterRect[1] + local[1], local[2], local[3])

def buildSolidBubbleMask(patch: np.ndarray) -> np.ndarray:
    if patch is None or patch.size == 0: return np.zeros((1,1), dtype=np.uint8)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(hsv, np.array([35, 60, 120]), np.array([70, 255, 255]))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel3)
    greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel5)
    contours, _ = cv2.findContours(greenMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) < patch.shape[0] * patch.shape[1] * 0.02: continue
        cv2.drawContours(solidMask, [c], -1, 255, cv2.FILLED)
    solidMask = cv2.morphologyEx(solidMask, cv2.MORPH_CLOSE, kernel5)
    if cv2.countNonZero(solidMask) == 0:
        solidMask[:] = 255
    return solidMask

def buildBubbleInteriorMask(bubbleMask: np.ndarray) -> np.ndarray:
    if bubbleMask is None or bubbleMask.size == 0: return np.zeros((1,1), dtype=np.uint8)
    interiorMask = bubbleMask.copy()
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    interiorMask = cv2.erode(interiorMask, kernel3)
    interiorMask = cv2.erode(interiorMask, kernel5)
    if cv2.countNonZero(interiorMask) == 0: return bubbleMask.copy()
    return interiorMask

def estimateBubbleFillColor(patch: np.ndarray, bubbleMask: np.ndarray) -> Tuple[float, float, float]:
    if patch is None or patch.size == 0 or bubbleMask is None or bubbleMask.size == 0: return (120, 230, 80)
    interiorMask = buildBubbleInteriorMask(bubbleMask)
    m = cv2.mean(patch, mask=interiorMask)
    return (m[0], m[1], m[2])

def estimateReferenceWhiteBubbleColor(img: np.ndarray) -> Tuple[float, float, float]:
    if img is None or img.size == 0: return DEFAULT_MIRRORED_BUBBLE_FILL
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    whiteMask = cv2.inRange(hsv, np.array([0, 0, 215]), np.array([180, 40, 255]))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_OPEN, kernel3)
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, kernel5)
    contours, _ = cv2.findContours(whiteMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    weightedSum = [0.0, 0.0, 0.0]
    totalWeight = 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 1200 or w * h > img.shape[1] * img.shape[0] * 0.08: continue
        if w < 45 or h < 28 or w > img.shape[1] * 0.55: continue
        if x > img.shape[1] * 0.60: continue
        if y < img.shape[0] * 0.08 or y > img.shape[0] * 0.92: continue
        ratio = float(w) / max(1, h)
        if ratio < 0.8 or ratio > 8.5: continue
        contourMask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(contourMask, [c], -1, 255, cv2.FILLED)
        sample = cv2.mean(img, mask=contourMask)
        weight = cv2.contourArea(c)
        weightedSum[0] += sample[0] * weight
        weightedSum[1] += sample[1] * weight
        weightedSum[2] += sample[2] * weight
        totalWeight += weight
    if totalWeight <= 0.0: return DEFAULT_MIRRORED_BUBBLE_FILL
    return (weightedSum[0] / totalWeight, weightedSum[1] / totalWeight, weightedSum[2] / totalWeight)

def detectBubbleContent(patch: np.ndarray, bubbleMask: np.ndarray, fillColor: Tuple[float, float, float]) -> BubbleContentResult:
    result = BubbleContentResult()
    if patch is None or patch.size == 0 or bubbleMask is None or bubbleMask.size == 0: return result
    interiorMask = buildBubbleInteriorMask(bubbleMask)
    bodyRect = bboxFromMaskPeakThreshold(interiorMask, 0.82, 0.82)
    if rect_area(bodyRect) == 0:
        bodyRect = bboxFromMaskProjection(interiorMask, 0.08, 0.08)
    if rect_area(bodyRect) == 0:
        bodyRect = (0, 0, patch.shape[1], patch.shape[0])
        
    bodyRect = insetRect(bodyRect, 1, 1, (patch.shape[1], patch.shape[0]))
    searchRect = insetRect(bodyRect, 2, 2, (patch.shape[1], patch.shape[0]))
    if rect_area(searchRect) == 0: searchRect = bodyRect
    
    fillMat = np.full(patch.shape, fillColor, dtype=patch.dtype)
    diffBgr = cv2.absdiff(patch, fillMat)
    ch = cv2.split(diffBgr)
    maxDiff = np.maximum(np.maximum(ch[0], ch[1]), ch[2])
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    bodyMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.rectangle(bodyMask, (bodyRect[0], bodyRect[1]), (bodyRect[0]+bodyRect[2], bodyRect[1]+bodyRect[3]), 255, cv2.FILLED)
    bodyMask = cv2.bitwise_and(bodyMask, interiorMask)
    
    searchMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.rectangle(searchMask, (searchRect[0], searchRect[1]), (searchRect[0]+searchRect[2], searchRect[1]+searchRect[3]), 255, cv2.FILLED)
    searchMask = cv2.bitwise_and(searchMask, bodyMask)
    
    _, looseMask = cv2.threshold(maxDiff, 8, 255, cv2.THRESH_BINARY)
    looseMask = cv2.bitwise_and(looseMask, searchMask)
    
    _, coreMask = cv2.threshold(maxDiff, 18, 255, cv2.THRESH_BINARY)
    coreMask = cv2.bitwise_and(coreMask, searchMask)
    coreMask = cv2.morphologyEx(coreMask, cv2.MORPH_CLOSE, kernel3)
    
    expandedCore = cv2.dilate(coreMask, kernel3)
    rawMask = cv2.bitwise_and(looseMask, expandedCore)
    rawMask = cv2.morphologyEx(rawMask, cv2.MORPH_CLOSE, kernel3)
    
    contours, _ = cv2.findContours(coreMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleanedMask = np.zeros(patch.shape[:2], dtype=np.uint8)
    unionRect = (0,0,0,0)
    hasContent = False
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 6: continue
        if w * h < 16: continue
        if w < 2 or h < 4: continue
        expanded = expandRect((x, y, w, h), 2, (patch.shape[1], patch.shape[0]))
        componentMask = np.zeros(patch.shape[:2], dtype=np.uint8)
        componentMask[expanded[1]:expanded[1]+expanded[3], expanded[0]:expanded[0]+expanded[2]] = rawMask[expanded[1]:expanded[1]+expanded[3], expanded[0]:expanded[0]+expanded[2]]
        componentMask = cv2.bitwise_and(componentMask, searchMask)
        if cv2.countNonZero(componentMask) == 0: continue
        cleanedMask = cv2.bitwise_or(cleanedMask, componentMask)
        unionRect = rect_or(unionRect, expanded) if hasContent else expanded
        hasContent = True
        
    if not hasContent:
        result.bodyRect = bodyRect
        result.bodyMaskFull = bodyMask
        result.searchMaskFull = searchMask
        result.coreMaskFull = coreMask
        result.looseMaskFull = looseMask
        return result
        
    eraseMask = cv2.dilate(cleanedMask, kernel3)
    eraseMask = cv2.morphologyEx(eraseMask, cv2.MORPH_CLOSE, kernel5)
    eraseMask = cv2.bitwise_and(eraseMask, searchMask)
    
    result.contentRect = unionRect
    result.visibleMask = cleanedMask[unionRect[1]:unionRect[1]+unionRect[3], unionRect[0]:unionRect[0]+unionRect[2]].copy()
    result.bodyRect = bodyRect
    result.pasteMaskFull = cleanedMask
    result.eraseMaskFull = eraseMask
    result.bodyMaskFull = bodyMask
    result.searchMaskFull = searchMask
    result.coreMaskFull = coreMask
    result.looseMaskFull = looseMask
    return result

def buildBubbleInfos(img: np.ndarray, bubbleRects: List[Tuple[int, int, int, int]]) -> List[BubbleInfo]:
    infos = []
    for r in bubbleRects:
        r = expandRect(r, 2, (img.shape[1], img.shape[0]))
        if r[2] <= 0 or r[3] <= 0: continue
        info = BubbleInfo()
        info.outerRect = r
        info.patch = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].copy()
        info.visibleMask = buildSolidBubbleMask(info.patch)
        info.fillColor = estimateBubbleFillColor(info.patch, info.visibleMask)
        
        content = detectBubbleContent(info.patch, info.visibleMask, info.fillColor)
        info.bodyRectInPatch = content.bodyRect
        info.bodyMaskFull = content.bodyMaskFull.copy() if content.bodyMaskFull is not None else None
        info.searchMaskFull = content.searchMaskFull.copy() if content.searchMaskFull is not None else None
        info.coreMaskFull = content.coreMaskFull.copy() if content.coreMaskFull is not None else None
        info.looseMaskFull = content.looseMaskFull.copy() if content.looseMaskFull is not None else None
        info.pasteMaskFull = content.pasteMaskFull.copy() if content.pasteMaskFull is not None else None
        info.eraseMaskFull = content.eraseMaskFull.copy() if content.eraseMaskFull is not None else None
        
        if rect_area(content.contentRect) > 0 and content.visibleMask is not None:
            info.contentRectInPatch = content.contentRect
            info.contentPatch = info.patch[content.contentRect[1]:content.contentRect[1]+content.contentRect[3], content.contentRect[0]:content.contentRect[0]+content.contentRect[2]].copy()
            info.contentMask = content.visibleMask.copy()
            info.hasContent = True
        infos.append(info)
    return infos

def mirroredOuterRectByContent(info: AvatarInfo, centerX: int) -> Tuple[int, int, int, int]:
    contentGlobal = contentRectInImage(info)
    targetContentX = 2 * centerX - (contentGlobal[0] + contentGlobal[2])
    targetOuterX = targetContentX - info.contentRectInPatch[0]
    return (targetOuterX, info.outerRect[1], info.outerRect[2], info.outerRect[3])

def mirroredBubbleRect(info: BubbleInfo, centerX: int) -> Tuple[int, int, int, int]:
    targetX = 2 * centerX - (info.outerRect[0] + info.outerRect[2])
    return (targetX, info.outerRect[1], info.outerRect[2], info.outerRect[3])

# --- NICKNAME RENDERING ---
def pickNicknameLatinFontPath() -> str:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc"
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return ""

def pickNicknameCjkFontPath() -> str:
    candidates = [
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return ""

def fontFaceIndexForNicknamePath(fontPath: str) -> int:
    return 11 if "PingFang.ttc" in fontPath else 0

def detectNicknameRectAboveBubble(img: np.ndarray, avatarRect: Tuple[int,int,int,int], bubbleRect: Tuple[int,int,int,int]) -> Tuple[bool, Tuple[int,int,int,int], Tuple[float,float,float]]:
    nicknameRect = (0,0,0,0)
    nicknameColor = (135.0, 135.0, 135.0)
    if img is None or img.size == 0 or rect_area(avatarRect) <= 0 or rect_area(bubbleRect) <= 0:
        return False, nicknameRect, nicknameColor
        
    x1 = min(img.shape[1] - 1, avatarRect[0] + avatarRect[2] + 4)
    x2 = min(img.shape[1], max(bubbleRect[0] + min(bubbleRect[2] + 80, img.shape[1] // 3), x1 + 24))
    y1 = max(0, avatarRect[1] - avatarRect[3] // 10)
    y2 = min(img.shape[0], bubbleRect[1] - 4)
    if x2 <= x1 or y2 <= y1: return False, nicknameRect, nicknameColor
    
    roiRect = (x1, y1, x2 - x1, y2 - y1)
    roi = img[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]].copy()
    
    bg = estimateBackgroundFromCorners(roi)
    bgGray = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    bgMat = np.full(roi.shape, bg, dtype=roi.dtype)
    diffBgr = cv2.absdiff(roi, bgMat)
    ch = cv2.split(diffBgr)
    maxDiff = np.maximum(np.maximum(ch[0], ch[1]), ch[2])
    
    _, diffMask = cv2.threshold(maxDiff, 8, 255, cv2.THRESH_BINARY)
    
    grayThr = max(0, int(bgGray - 12.0))
    _, darkMask = cv2.threshold(gray, grayThr, 255, cv2.THRESH_BINARY_INV)
    
    lowSatMask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 85, 225]))
    
    mask = cv2.bitwise_and(darkMask, lowSatMask)
    mask = cv2.bitwise_and(mask, diffMask)
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mergedMask = np.zeros(mask.shape, dtype=np.uint8)
    unionRect = (0,0,0,0)
    found = False
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 6: continue
        if w < 2 or h < 7: continue
        if h > roi.shape[0]: continue
        cv2.drawContours(mergedMask, [c], -1, 255, cv2.FILLED)
        unionRect = rect_or(unionRect, (x, y, w, h)) if found else (x, y, w, h)
        found = True
        
    if not found or unionRect[2] < 10 or unionRect[3] < 8:
        return False, nicknameRect, nicknameColor
        
    nicknameRect = (roiRect[0] + unionRect[0], roiRect[1] + unionRect[1], unionRect[2], unionRect[3])
    roi_crop = roi[unionRect[1]:unionRect[1]+unionRect[3], unionRect[0]:unionRect[0]+unionRect[2]]
    mask_crop = mergedMask[unionRect[1]:unionRect[1]+unionRect[3], unionRect[0]:unionRect[0]+unionRect[2]]
    color = cv2.mean(roi_crop, mask=mask_crop)
    return True, nicknameRect, (color[0], color[1], color[2])

def estimateNicknameReferenceStyle(img: np.ndarray, avatarInfos: List[AvatarInfo], bubbleInfos: List[BubbleInfo]) -> NicknameReferenceStyle:
    style = NicknameReferenceStyle()
    style.cjkFontPath = pickNicknameCjkFontPath()
    style.cjkFontIndex = fontFaceIndexForNicknamePath(style.cjkFontPath)
    style.latinFontPath = style.cjkFontPath
    style.latinFontIndex = style.cjkFontIndex
    
    avatarSize = 85
    for info in avatarInfos:
        if info.contentRectInPatch[2] > 0:
            avatarSize = min(info.contentRectInPatch[2], info.contentRectInPatch[3])
            break
            
    scale = float(avatarSize) / 85.0
    
    style.valid = True
    style.color = (125.0, 125.0, 125.0)
    style.referenceAvatarSize = avatarSize
    style.fontHeight = max(16, int(round(21.0 * scale)))
    style.latinFontHeight = style.fontHeight
    
    style.textXOffsetFromAvatarRight = int(round(26.0 * scale))
    style.textTopOffsetFromAvatarTop = int(round(5.0 * scale))
    style.bubbleXOffsetFromAvatarRight = int(round(14.0 * scale))
    style.nicknameToBubbleGap = int(round(3.0 * scale))
    style.bubbleYOffsetFromAvatarTop = int(round(28.0 * scale))
    style.blockGapY = max(6, int(round(10.0 * scale)))
    
    style.referenceBubbleX = -1
    style.referenceNicknameX = -1
    
    return style

# Since we use PIL for fonts, we implement createNicknameTextRenderer and friends
class PythonNicknameTextRenderer:
    def __init__(self, style: NicknameReferenceStyle):
        self.latinPath = style.latinFontPath
        self.latinIndex = style.latinFontIndex
        self.cjkPath = style.cjkFontPath
        self.cjkIndex = style.cjkFontIndex
        self.cache = {}
        
    def get_font(self, asciiRun: bool, size: int) -> Optional[ImageFont.FreeTypeFont]:
        path = self.latinPath if asciiRun else self.cjkPath
        idx = self.latinIndex if asciiRun else self.cjkIndex
        if not path or not os.path.exists(path):
            path = self.cjkPath if asciiRun else self.latinPath
            idx = self.cjkIndex if asciiRun else self.latinIndex
        if not path or not os.path.exists(path): return None
        key = f"{path}_{idx}_{size}"
        if key not in self.cache:
            try:
                self.cache[key] = ImageFont.truetype(path, size, index=idx)
            except:
                return None
        return self.cache[key]

def createNicknameTextRenderer(style: NicknameReferenceStyle) -> PythonNicknameTextRenderer:
    return PythonNicknameTextRenderer(style)

def splitNicknameRuns(text: str) -> List[Tuple[str, bool]]:
    runs = []
    if not text: return runs
    start = 0
    currentAscii = ord(text[0]) < 0x80
    for i in range(1, len(text)):
        asciiFlg = ord(text[i]) < 0x80
        if asciiFlg != currentAscii:
            runs.append((text[start:i], currentAscii))
            start = i
            currentAscii = asciiFlg
    runs.append((text[start:], currentAscii))
    return runs

def nicknameRunFontHeight(style: NicknameReferenceStyle, asciiRun: bool) -> int:
    return max(1, style.latinFontHeight) if asciiRun else max(1, style.fontHeight)

def measureNicknameRunAtHeight(text: str, asciiRun: bool, fontHeight: int, renderer: PythonNicknameTextRenderer) -> Tuple[int, int]:
    font = renderer.get_font(asciiRun, fontHeight)
    if font:
        img_pil = Image.new('L', (1, 1), 0)
        draw = ImageDraw.Draw(img_pil)
        w_total = 0
        h_max = fontHeight
        last_c = ''
        kern_map = { ('T', 'o'): -3.0 }
        kern_scale = fontHeight / 31.0
        for c in text:
            if (last_c, c) in kern_map:
                w_total += int(round(kern_map[(last_c, c)] * kern_scale))
            try:
                w_total += int(draw.textlength(c, font=font))
            except AttributeError:
                w_total += font.getsize(c)[0]
            last_c = c
        return (w_total, h_max)
    # Fallback OpenCV
    scale = max(0.45, fontHeight / 28.0)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return (w, h)

def measureNicknameRun(text: str, asciiRun: bool, style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer) -> Tuple[int, int]:
    return measureNicknameRunAtHeight(text, asciiRun, nicknameRunFontHeight(style, asciiRun), renderer)

def measureNicknameText(text: str, style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer) -> Tuple[int, int]:
    totalWidth = 0
    maxHeight = 0
    for textRun, asciiRun in splitNicknameRuns(text):
        w, h = measureNicknameRun(textRun, asciiRun, style, renderer)
        totalWidth += w
        maxHeight = max(maxHeight, h)
    return (totalWidth, maxHeight)

def measureNicknameInkHeight(text: str, asciiRun: bool, fontHeight: int, renderer: PythonNicknameTextRenderer) -> int:
    if not text: return 0
    w, h = measureNicknameRunAtHeight(text, asciiRun, fontHeight, renderer)
    if w <= 0 or h <= 0: return 0
    
    font = renderer.get_font(asciiRun, fontHeight)
    if font:
        img_pil = Image.new('L', (w + 24, h + 24), 0)
        draw = ImageDraw.Draw(img_pil)
        draw.text((12, 12), text, font=font, fill=255)
        nz = np.nonzero(np.array(img_pil))
        if len(nz[0]) > 0:
            return int(np.max(nz[0]) - np.min(nz[0]) + 1)
    return h

def drawNicknameText(img: np.ndarray, text: str, textRect: Tuple[int,int,int,int], style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer):
    if img is None or not text or rect_area(textRect) <= 0: return
    
    totalW, maxH = measureNicknameText(text, style, renderer)
    cursorX = textRect[0]
    baselineY = textRect[1]
    
    for textRun, asciiRun in splitNicknameRuns(text):
        font = renderer.get_font(asciiRun, nicknameRunFontHeight(style, asciiRun))
        w, _ = measureNicknameRun(textRun, asciiRun, style, renderer)
        if font:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            rgb_color = (int(style.color[2]), int(style.color[1]), int(style.color[0])) # PIL uses RGB
            
            kern_map = { ('T', 'o'): -3.0 }
            kern_scale = nicknameRunFontHeight(style, asciiRun) / 31.0
            last_c = ''
            for c in textRun:
                if (last_c, c) in kern_map:
                    cursorX += int(round(kern_map[(last_c, c)] * kern_scale))
                draw.text((cursorX, baselineY), c, font=font, fill=rgb_color)
                try:
                    cursorX += int(draw.textlength(c, font=font))
                except AttributeError:
                    cursorX += font.getsize(c)[0]
                last_c = c
                
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            scale = max(0.45, nicknameRunFontHeight(style, asciiRun) / 28.0)
            cv2.putText(img, textRun, (cursorX, baselineY + int(scale * 22)), cv2.FONT_HERSHEY_SIMPLEX, scale, style.color, 1, cv2.LINE_AA)
        cursorX += w

def measureNicknameInkBounds(text: str, style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer, lowercaseOnly: bool, alphaThreshold: int) -> Tuple[int, int, int, int]:
    if not text: return (0, 0, 0, 0)
    totalW, maxH = measureNicknameText(text, style, renderer)
    if totalW <= 0 or maxH <= 0: return (0, 0, 0, 0)
    
    pad = 20
    img_pil = Image.new('L', (totalW + pad * 2, maxH + pad * 2), 0)
    draw = ImageDraw.Draw(img_pil)
    
    cursorX = pad
    baselineY = pad
    
    for textRun, asciiRun in splitNicknameRuns(text):
        w, _ = measureNicknameRun(textRun, asciiRun, style, renderer)
        font = renderer.get_font(asciiRun, nicknameRunFontHeight(style, asciiRun))
        
        drawText = textRun
        if lowercaseOnly:
            if not asciiRun:
                cursorX += w
                continue
            chars = []
            hasLowercase = False
            for ch in drawText:
                if 'a' <= ch <= 'z':
                    chars.append(ch)
                    hasLowercase = True
                else:
                    chars.append(' ')
            if not hasLowercase:
                cursorX += w
                continue
            drawText = "".join(chars)
            
        if font:
            kern_map = { ('T', 'o'): -3.0 }
            kern_scale = nicknameRunFontHeight(style, asciiRun) / 31.0
            last_c = ''
            cx = cursorX
            for c in drawText:
                if (last_c, c) in kern_map:
                    cx += int(round(kern_map[(last_c, c)] * kern_scale))
                draw.text((cx, baselineY), c, font=font, fill=255)
                try:
                    cx += int(draw.textlength(c, font=font))
                except AttributeError:
                    cx += font.getsize(c)[0]
                last_c = c
        else:
            scale = max(0.45, nicknameRunFontHeight(style, asciiRun) / 28.0)
            img_cv = np.array(img_pil)
            cv2.putText(img_cv, drawText, (cursorX, baselineY + int(scale * 22)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
            img_pil = Image.fromarray(img_cv)
            draw = ImageDraw.Draw(img_pil)
        cursorX += w
        
    mask = np.array(img_pil)
    _, strongMask = cv2.threshold(mask, alphaThreshold, 255, cv2.THRESH_BINARY)
    bounds = boundingRectOfMask(strongMask)
    if rect_area(bounds) <= 0:
        bounds = boundingRectOfMask(mask)
    if rect_area(bounds) <= 0: return (0, 0, 0, 0)
    return (bounds[0] - pad, bounds[1] - pad, bounds[2], bounds[3])

def calibrateNicknameLatinFontHeight(style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer) -> int:
    return style.fontHeight

def calibrateNicknameStyleHeights(style: NicknameReferenceStyle, renderer: PythonNicknameTextRenderer):
    style.latinFontHeight = calibrateNicknameLatinFontHeight(style, renderer)

# --- CORE PROCESSING ---
def adaptTextPatchToTargetFill(info: BubbleInfo, targetFillColor: Tuple[float, float, float]) -> np.ndarray:
    if info.contentPatch is None or info.contentMask is None: return np.zeros((1,1), dtype=np.uint8)
    adjusted = info.contentPatch.copy()
    for y in range(adjusted.shape[0]):
        for x in range(adjusted.shape[1]):
            if info.contentMask[y, x] == 0: continue
            alpha = 0.0
            for c in range(3):
                srcFill = float(info.fillColor[c])
                srcPixel = float(info.contentPatch[y, x, c])
                denom = max(1.0, srcFill)
                channelAlpha = (srcFill - srcPixel) / denom
                alpha = max(alpha, min(1.0, max(0.0, channelAlpha)))
            for c in range(3):
                target = float(targetFillColor[c])
                v = (1.0 - alpha) * target
                adjusted[y, x, c] = int(max(0, min(255, v)))
    return adjusted

def adaptBubblePatchToTargetFill(patch: np.ndarray, bubbleMask: np.ndarray, sourceFillColor: Tuple[float, float, float], targetFillColor: Tuple[float, float, float]) -> np.ndarray:
    if patch is None or patch.size == 0: return np.zeros((1,1), dtype=np.uint8)
    adjusted = patch.copy()
    sourceLuma = 0.114 * sourceFillColor[0] + 0.587 * sourceFillColor[1] + 0.299 * sourceFillColor[2]
    
    for y in range(adjusted.shape[0]):
        for x in range(adjusted.shape[1]):
            if bubbleMask is not None and bubbleMask.size > 0 and bubbleMask[y, x] == 0: continue
            srcLuma = 0.114 * patch[y, x, 0] + 0.587 * patch[y, x, 1] + 0.299 * patch[y, x, 2]
            shadeDelta = srcLuma - sourceLuma
            for c in range(3):
                target = targetFillColor[c] + shadeDelta
                adjusted[y, x, c] = int(max(0, min(255, target)))
    return adjusted

def bubbleBorderColor(fillColor: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (max(0.0, fillColor[0] - 12.0), max(0.0, fillColor[1] - 12.0), max(0.0, fillColor[2] - 12.0))

def renderCrispBubbleBase(bubbleMask: np.ndarray, fillColor: Tuple[float, float, float]) -> np.ndarray:
    if bubbleMask is None or bubbleMask.size == 0: return np.zeros((1,1), dtype=np.uint8)
    cleanedMask = bubbleMask.copy()
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_OPEN, kernel3)
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_CLOSE, kernel5)
    
    contours, _ = cv2.findContours(cleanedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidMask = np.zeros(cleanedMask.shape, dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) < bubbleMask.shape[0] * bubbleMask.shape[1] * 0.02: continue
        cv2.drawContours(solidMask, [c], -1, 255, cv2.FILLED, cv2.LINE_8)
    if cv2.countNonZero(solidMask) == 0:
        solidMask = cleanedMask
        
    bubble = np.zeros((solidMask.shape[0], solidMask.shape[1], 3), dtype=np.uint8)
    bubble[solidMask > 0] = fillColor
    
    innerMask = cv2.erode(solidMask, kernel3)
    borderMask = cv2.subtract(solidMask, innerMask)
    if cv2.countNonZero(borderMask) > 0:
        bubble[borderMask > 0] = bubbleBorderColor(fillColor)
    return bubble

def readableMirroredContentRect(info: BubbleInfo) -> Tuple[int, int, int, int]:
    pw = info.patch.shape[1] if info.patch is not None else 0
    def mirroredRectInPatch(r: Tuple[int, int, int, int], patchWidth: int) -> Tuple[int, int, int, int]:
        return (patchWidth - (r[0] + r[2]), r[1], r[2], r[3])
    
    mirroredContent = mirroredRectInPatch(info.contentRectInPatch, pw)
    if rect_area(info.bodyRectInPatch) <= 0 or rect_area(info.contentRectInPatch) <= 0:
        return mirroredContent
        
    mirroredBody = mirroredRectInPatch(info.bodyRectInPatch, pw)
    leftPadding = max(0, info.contentRectInPatch[0] - info.bodyRectInPatch[0])
    topPadding = max(0, info.contentRectInPatch[1] - info.bodyRectInPatch[1])
    
    targetX = mirroredBody[0] + leftPadding
    targetY = mirroredBody[1] + topPadding
    
    maxX = max(0, mirroredBody[0] + mirroredBody[2] - info.contentRectInPatch[2])
    maxY = max(0, mirroredBody[1] + mirroredBody[3] - info.contentRectInPatch[3])
    targetX = min(max(targetX, 0), maxX)
    targetY = min(max(targetY, 0), maxY)
    
    return (targetX, targetY, info.contentRectInPatch[2], info.contentRectInPatch[3])

def buildMirroredBubblePatch(info: BubbleInfo, keepTextReadable: bool, targetFillColor: Tuple[float, float, float]) -> np.ndarray:
    flippedPatch = cv2.flip(info.patch, 1)
    flippedBubbleMask = cv2.flip(info.visibleMask, 1) if info.visibleMask is not None else None
    
    if flippedBubbleMask is not None and flippedBubbleMask.size > 0:
        crispBubble = renderCrispBubbleBase(flippedBubbleMask, targetFillColor)
        if crispBubble is not None and crispBubble.size > 0 and len(crispBubble.shape) == 3:
            flippedPatch = crispBubble
        else:
            recoloredBubble = adaptBubblePatchToTargetFill(flippedPatch, flippedBubbleMask, info.fillColor, targetFillColor)
            if recoloredBubble is not None and recoloredBubble.size > 0:
                flippedPatch = recoloredBubble
                
    if not keepTextReadable or not info.hasContent or info.contentPatch is None or info.contentMask is None:
        return flippedPatch
        
    mirroredContent = readableMirroredContentRect(info)
    if rect_area(rect_and(mirroredContent, (0, 0, flippedPatch.shape[1], flippedPatch.shape[0]))) != rect_area(mirroredContent):
        return flippedPatch
        
    adjustedTextPatch = adaptTextPatchToTargetFill(info, targetFillColor)
    if adjustedTextPatch is None or adjustedTextPatch.size == 0:
        return flippedPatch
        
    for y in range(adjustedTextPatch.shape[0]):
        for x in range(adjustedTextPatch.shape[1]):
            if info.contentMask[y, x] > 0:
                ty = mirroredContent[1] + y
                tx = mirroredContent[0] + x
                if 0 <= ty < flippedPatch.shape[0] and 0 <= tx < flippedPatch.shape[1]:
                    flippedPatch[ty, tx] = adjustedTextPatch[y, x]
    return flippedPatch

def findReferenceGrayPatchRect(img: np.ndarray) -> Tuple[int, int, int, int]:
    if img is None or img.size == 0: return (0,0,0,0)
    patchW = min(max(72, img.shape[1] // 9), max(72, img.shape[1] // 4))
    patchH = min(max(72, img.shape[0] // 10), max(72, img.shape[0] // 5))
    
    xStart = min(max(0, int(img.shape[1] * 0.56)), max(0, img.shape[1] - patchW))
    xEnd = max(xStart, img.shape[1] - patchW - max(12, img.shape[1] // 18))
    yStart = min(max(0, int(img.shape[0] * 0.15)), max(0, img.shape[0] - patchH))
    yEnd = max(yStart, min(max(0, img.shape[0] - patchH), int(img.shape[0] * 0.42)))
    
    stepX = max(12, patchW // 4)
    stepY = max(12, patchH // 4)
    
    bestRect = (0,0,0,0)
    bestScore = 1e18
    
    for y in range(yStart, yEnd + 1, stepY):
        for x in range(xStart, xEnd + 1, stepX):
            r = (x, y, patchW, patchH)
            if rect_area(rect_and(r, (0, 0, img.shape[1], img.shape[0]))) != rect_area(r): continue
            
            roi = img[y:y+patchH, x:x+patchW]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            meanGray, stdGray = cv2.meanStdDev(gray)
            edges = cv2.Canny(gray, 30, 90)
            edgeRatio = float(cv2.countNonZero(edges)) / max(1, rect_area(r))
            
            meanHsv = cv2.mean(hsv)
            saturation = meanHsv[1]
            brightnessPenalty = abs(meanGray[0][0] - 236.0)
            
            score = stdGray[0][0] * 4.0 + edgeRatio * 2500.0 + saturation * 0.8 + brightnessPenalty * 0.4
            if score < bestScore:
                bestScore = score
                bestRect = r
                
    if rect_area(bestRect) > 0: return bestRect
    fallbackX = max(0, img.shape[1] - patchW - max(12, img.shape[1] // 18))
    fallbackY = min(max(0, int(img.shape[0] * 0.18)), max(0, img.shape[0] - patchH))
    return rect_and((fallbackX, fallbackY, patchW, patchH), (0, 0, img.shape[1], img.shape[0]))

def coverRectWithReferencePatch(dst: np.ndarray, referencePatch: np.ndarray, targetRect: Tuple[int, int, int, int]):
    r = rect_and(targetRect, (0, 0, dst.shape[1], dst.shape[0]))
    if rect_area(r) <= 0 or referencePatch is None or referencePatch.size == 0: return
    resized = cv2.resize(referencePatch, (r[2], r[3]), interpolation=cv2.INTER_CUBIC)
    resized = cv2.GaussianBlur(resized, (15, 15), 0)
    dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = resized

def shiftImageContentDown(img: np.ndarray, cutoffY: int, shiftY: int, referencePatch: np.ndarray):
    if img is None or img.size == 0 or shiftY <= 0: return
    if cutoffY < 0 or cutoffY >= img.shape[0]: return
    shiftY = min(shiftY, img.shape[0] - cutoffY)
    if shiftY <= 0: return
    
    original = img.copy()
    movableHeight = img.shape[0] - cutoffY - shiftY
    if movableHeight > 0:
        img[cutoffY+shiftY:cutoffY+shiftY+movableHeight, :] = original[cutoffY:cutoffY+movableHeight, :]
    coverRectWithReferencePatch(img, referencePatch, (0, cutoffY, img.shape[1], shiftY))

def removeOriginalObjectsByInpaint(input_img: np.ndarray, avatarInfos: List[AvatarInfo], bubbleInfos: List[BubbleInfo]) -> np.ndarray:
    if input_img is None or input_img.size == 0: return np.zeros((1,1), dtype=np.uint8)
    cleaned = input_img.copy()
    refRect = findReferenceGrayPatchRect(input_img)
    referencePatch = input_img[refRect[1]:refRect[1]+refRect[3], refRect[0]:refRect[0]+refRect[2]].copy() if rect_area(refRect) > 0 else np.full((96, 96, 3), (238, 238, 238), dtype=input_img.dtype)
    
    for info in bubbleInfos:
        cover = expandRectAsym(info.outerRect, 10, 8, 22, 8, (input_img.shape[1], input_img.shape[0]))
        coverRectWithReferencePatch(cleaned, referencePatch, cover)
    for info in avatarInfos:
        cover = expandRectAsym(info.outerRect, 8, 8, 12, 8, (input_img.shape[1], input_img.shape[0]))
        coverRectWithReferencePatch(cleaned, referencePatch, cover)
    return cleaned

def pastePatchWithMaskSafe(dst: np.ndarray, patch: np.ndarray, mask: np.ndarray, targetRect: Tuple[int, int, int, int]):
    bounds = (0, 0, dst.shape[1], dst.shape[0])
    clippedTarget = rect_and(targetRect, bounds)
    if rect_area(clippedTarget) <= 0: return
    
    sx = clippedTarget[0] - targetRect[0]
    sy = clippedTarget[1] - targetRect[1]
    
    dst_roi = dst[clippedTarget[1]:clippedTarget[1]+clippedTarget[3], clippedTarget[0]:clippedTarget[0]+clippedTarget[2]]
    patch_roi = patch[sy:sy+clippedTarget[3], sx:sx+clippedTarget[2]]
    mask_roi = mask[sy:sy+clippedTarget[3], sx:sx+clippedTarget[2]]
    
    idx = mask_roi > 0
    if len(mask_roi.shape) == 2 and len(dst_roi.shape) == 3:
        dst_roi[idx] = patch_roi[idx]
    else:
        dst_roi[idx] = patch_roi[idx]

def drawCenterLine(img: np.ndarray):
    centerX = img.shape[1] // 2
    outerThickness = max(4, img.shape[1] // 300)
    innerThickness = max(2, outerThickness // 2)
    cv2.line(img, (centerX, 0), (centerX, img.shape[0] - 1), (0, 0, 0), outerThickness, cv2.LINE_AA)
    cv2.line(img, (centerX, 0), (centerX, img.shape[0] - 1), (0, 255, 255), innerThickness, cv2.LINE_AA)
    cv2.circle(img, (centerX, 30), 8, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
    cv2.putText(img, "CENTER", (max(10, centerX - 45), 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "CENTER", (max(10, centerX - 45), 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

def processImage(input_img: np.ndarray, outputPath: str, options: ProcessOptions) -> np.ndarray:
    if input_img is None or input_img.size == 0: return np.zeros((1,1), dtype=np.uint8)
    bubbles = findRightGreenBubbles(input_img)
    avatarRects = findAllAvatars(input_img, bubbles)
    mirroredBubbleFillColor = estimateReferenceWhiteBubbleColor(input_img)
    
    bubbleInfos = buildBubbleInfos(input_img, bubbles)
    avatarInfos = buildAvatarInfos(input_img, avatarRects)
    
    nicknameStyle = None
    nicknameRenderer = None
    
    if options.withNickname:
        nicknameStyle = estimateNicknameReferenceStyle(input_img, avatarInfos, bubbleInfos)
        nicknameRenderer = createNicknameTextRenderer(nicknameStyle)
        calibrateNicknameStyleHeights(nicknameStyle, nicknameRenderer)
        
    centerX = input_img.shape[1] // 2
    print(f"Mirrored bubble fill=({int(mirroredBubbleFillColor[0])},{int(mirroredBubbleFillColor[1])},{int(mirroredBubbleFillColor[2])})")
    print(f"Center X = {centerX}")
    print(f"Bubbles: {len(bubbleInfos)}")
    print(f"Avatars: {len(avatarInfos)}")
    
    if options.withNickname:
        latinInkHeight = measureNicknameInkHeight("en", True, nicknameStyle.latinFontHeight, nicknameRenderer)
        print(f"Nickname mode: yes ({options.nickname})")
    else:
        print("Nickname mode: no")
        
    output = removeOriginalObjectsByInpaint(input_img, avatarInfos, bubbleInfos)
    grayRefRect = findReferenceGrayPatchRect(input_img)
    if rect_area(grayRefRect) > 0:
        grayReferencePatch = input_img[grayRefRect[1]:grayRefRect[1]+grayRefRect[3], grayRefRect[0]:grayRefRect[0]+grayRefRect[2]].copy()
    else:
        grayReferencePatch = np.full((96, 96, 3), (238, 238, 238), dtype=input_img.dtype)
        
    targetBubbleRects = [(0,0,0,0)] * len(bubbleInfos)
    hasTargetBubble = [0] * len(bubbleInfos)
    targetAvatarRects = [(0,0,0,0)] * len(avatarInfos)
    hasTargetAvatar = [0] * len(avatarInfos)
    targetNicknameRects = [(0,0,0,0)] * len(bubbleInfos)
    hasTargetNickname = [0] * len(bubbleInfos)
    
    nicknameInkBounds = (0,0,0,0)
    nicknameLowercaseBounds = (0,0,0,0)
    nicknameLowercaseSampleBounds = (0,0,0,0)
    nicknameHasLowercase = False
    
    mirroredBubbleFillColor = (255.0, 255.0, 255.0)
    
    groups = [] 
    a_idx = 0
    for b_idx, b_info in enumerate(bubbleInfos):
        b_y = b_info.outerRect[1]
        if a_idx + 1 < len(avatarInfos):
            next_a_y = avatarInfos[a_idx + 1].outerRect[1]
            if next_a_y - 20 <= b_y:
                a_idx += 1
        if a_idx < len(avatarInfos):
            if len(groups) == 0 or groups[-1]["avatar_idx"] != a_idx:
                groups.append({"avatar_idx": a_idx, "bubble_indices": []})
            groups[-1]["bubble_indices"].append(b_idx)
            
    if options.withNickname and options.nickname:
        nicknameHasLowercase = hasAsciiLowercase(options.nickname)
        nicknameInkBounds = measureNicknameInkBounds(options.nickname, nicknameStyle, nicknameRenderer, False, 72)
        nicknameLowercaseBounds = measureNicknameInkBounds(options.nickname, nicknameStyle, nicknameRenderer, True, 112)
        nicknameLowercaseSampleBounds = measureNicknameInkBounds("xneo", nicknameStyle, nicknameRenderer, False, 112)
        
    previousBottom = -1000000
    
    for group in groups:
        a_i = group["avatar_idx"]
        b_indices = group["bubble_indices"]
        
        targetAvatar = mirroredOuterRectByContent(avatarInfos[a_i], centerX)
        targetAvatarContent = contentRectInTargetImage(avatarInfos[a_i], targetAvatar)
        anchorRight = targetAvatarContent[0] + targetAvatarContent[2]
        
        avatarSize = min(avatarInfos[a_i].contentRectInPatch[2], avatarInfos[a_i].contentRectInPatch[3])
        if avatarSize <= 0: avatarSize = 85
        scale = float(avatarSize) / 85.0
        bubbleXOffset = int(round(14.0 * scale))
        
        bubble_y_push = 0
        first_bubble = True
        
        for b_i in b_indices:
            b_info = bubbleInfos[b_i]
            targetBubble = mirroredBubbleRect(b_info, centerX)
            flippedBubbleMask = cv2.flip(b_info.visibleMask, 1) if b_info.visibleMask is not None else None
            flippedBubbleTipX = leftmostMaskXInBand(flippedBubbleMask)
            
            b_x = anchorRight + bubbleXOffset - max(0, flippedBubbleTipX)
            
            if first_bubble:
                if options.withNickname and options.nickname:
                    nicknameSize = measureNicknameText(options.nickname, nicknameStyle, nicknameRenderer)
                    nicknameVisualLeft = nicknameInkBounds[0] if rect_area(nicknameInkBounds) > 0 else 0
                    if nicknameHasLowercase:
                        nicknameAlignTop = nicknameLowercaseBounds[1] if rect_area(nicknameLowercaseBounds) > 0 else (nicknameLowercaseSampleBounds[1] if rect_area(nicknameLowercaseSampleBounds) > 0 else max(1, int(round(nicknameStyle.latinFontHeight * 0.58))))
                    else:
                        nicknameAlignTop = nicknameInkBounds[1] if rect_area(nicknameInkBounds) > 0 else max(1, int(round(nicknameStyle.fontHeight * 0.38)))
        
                    nicknameTopY = targetAvatarContent[1] + nicknameStyle.textTopOffsetFromAvatarTop - nicknameAlignTop
                    nx = anchorRight + nicknameStyle.textXOffsetFromAvatarRight - nicknameVisualLeft
                    nicknameRect = (nx, nicknameTopY, nicknameSize[0], max(1, nicknameSize[1]))
                    nicknameInkBottom = nicknameRect[1] + (nicknameInkBounds[1] + nicknameInkBounds[3] if rect_area(nicknameInkBounds) > 0 else int(round(nicknameStyle.fontHeight * 2.0)))
                    
                    blockTop = min(targetAvatar[1], nicknameRect[1])
                    shiftY_block = 0
                    if blockTop < previousBottom + nicknameStyle.blockGapY:
                        shiftY_block = previousBottom + nicknameStyle.blockGapY - blockTop
                        
                    targetAvatar = (targetAvatar[0], targetAvatar[1] + shiftY_block, targetAvatar[2], targetAvatar[3])
                    targetAvatarContent = (targetAvatarContent[0], targetAvatarContent[1] + shiftY_block, targetAvatarContent[2], targetAvatarContent[3])
                    nicknameRect = (nicknameRect[0], nicknameRect[1] + shiftY_block, nicknameRect[2], nicknameRect[3])
                    nicknameInkBottom += shiftY_block
                    
                    targetNicknameRects[b_i] = nicknameRect
                    hasTargetNickname[b_i] = 1
                    
                    new_by = max(targetAvatarContent[1] + nicknameStyle.bubbleYOffsetFromAvatarTop, nicknameInkBottom + nicknameStyle.nicknameToBubbleGap)
                    print(f"DEBUG: AvatarTop={targetAvatarContent[1]}, NicknameInkTop={nicknameTopY}, NicknameInkBottom={nicknameInkBottom}, new_by={new_by}, shiftY_block={shiftY_block}")
                    bubble_y_push = new_by - (targetBubble[1] + shiftY_block)
                    
                    targetAvatarRects[a_i] = targetAvatar
                    hasTargetAvatar[a_i] = 1
                    previousBottom = max(previousBottom, nicknameInkBottom)
                else:
                    blockTop = targetAvatar[1]
                    shiftY_block = 0
                    if blockTop < previousBottom + 10:
                        shiftY_block = previousBottom + 10 - blockTop
                        
                    targetAvatar = (targetAvatar[0], targetAvatar[1] + shiftY_block, targetAvatar[2], targetAvatar[3])
                    targetAvatarContent = (targetAvatarContent[0], targetAvatarContent[1] + shiftY_block, targetAvatarContent[2], targetAvatarContent[3])
                    
                    new_by = targetAvatarContent[1]
                    bubble_y_push = new_by - (targetBubble[1] + shiftY_block)
                    
                    targetAvatarRects[a_i] = targetAvatar
                    hasTargetAvatar[a_i] = 1
            
            shiftY_block = (targetAvatar[1] - mirroredOuterRectByContent(avatarInfos[a_i], centerX)[1])
            targetBubble = (b_x, targetBubble[1] + shiftY_block + bubble_y_push, targetBubble[2], targetBubble[3])
            
            targetBubbleRects[b_i] = targetBubble
            hasTargetBubble[b_i] = 1
            
            previousBottom = max(previousBottom, targetBubble[1] + targetBubble[3])
            first_bubble = False
            
        previousBottom = max(previousBottom, targetAvatar[1] + targetAvatar[3])
        
    lastGroup = groups[-1] if len(groups) > 0 else None
    if lastGroup is not None:
        a_i_last = lastGroup["avatar_idx"]
        b_i_last = lastGroup["bubble_indices"][-1]
        oldAvatarRect = mirroredOuterRectByContent(avatarInfos[a_i_last], centerX)
        oldBubbleRect = mirroredBubbleRect(bubbleInfos[b_i_last], centerX)
        oldBlockBottom = max(oldAvatarRect[1] + oldAvatarRect[3], oldBubbleRect[1] + oldBubbleRect[3])
        newBlockBottom = max(targetAvatarRects[a_i_last][1] + targetAvatarRects[a_i_last][3], targetBubbleRects[b_i_last][1] + targetBubbleRects[b_i_last][3])
        extraShift = max(0, newBlockBottom - oldBlockBottom + (nicknameStyle.blockGapY if nicknameStyle else 10))
        if extraShift > 0:
            shiftImageContentDown(output, oldBlockBottom, extraShift, grayReferencePatch)
                
    for i in range(len(bubbleInfos)):
        targetBubble = targetBubbleRects[i] if hasTargetBubble[i] else mirroredBubbleRect(bubbleInfos[i], centerX)
        mirroredPatch = buildMirroredBubblePatch(bubbleInfos[i], KEEP_BUBBLE_TEXT_READABLE, mirroredBubbleFillColor)
        flippedMask = cv2.flip(bubbleInfos[i].visibleMask, 1) if bubbleInfos[i].visibleMask is not None else None
        pastePatchWithMaskSafe(output, mirroredPatch, flippedMask, targetBubble)
        
    for i in range(len(avatarInfos)):
        targetOuter = targetAvatarRects[i] if hasTargetAvatar[i] else mirroredOuterRectByContent(avatarInfos[i], centerX)
        pastePatchWithMaskSafe(output, avatarInfos[i].patch, avatarInfos[i].visibleMask, targetOuter)
        
    if options.withNickname and options.nickname:
        for i in range(len(targetNicknameRects)):
            if not hasTargetNickname[i]: continue
            drawNicknameText(output, options.nickname, targetNicknameRects[i], nicknameStyle, nicknameRenderer)
            
    if DRAW_DEBUG:
        drawCenterLine(output)
        
    return output

# --- CLI ENTRYPOINT ---
if __name__ == "__main__":
    options = ProcessOptions()
    inputPath = ""
    outputPath = ""
    
    if len(sys.argv) == 3:
        inputPath = sys.argv[1]
        outputPath = sys.argv[2]
    elif len(sys.argv) == 4 and sys.argv[1] == "nonickname":
        inputPath = sys.argv[2]
        outputPath = sys.argv[3]
    elif len(sys.argv) == 5 and sys.argv[1] == "yesnickname":
        options.withNickname = True
        options.nickname = sys.argv[2]
        inputPath = sys.argv[3]
        outputPath = sys.argv[4]
    else:
        print(f"Usage: {sys.argv[0]} <input> <output>")
        print(f"   or: {sys.argv[0]} nonickname <input> <output>")
        print(f"   or: {sys.argv[0]} yesnickname <nickname> <input> <output>")
        sys.exit(1)
        
    inputImage = cv2.imread(inputPath, cv2.IMREAD_COLOR)
    if inputImage is None:
        print(f"Failed to load image: {inputPath}")
        sys.exit(1)
        
    outputImage = processImage(inputImage, outputPath, options)
    if outputImage is None or outputImage.size == 0:
        print("Processing failed")
        sys.exit(1)
        
    if not cv2.imwrite(outputPath, outputImage):
        print(f"Failed to save image: {outputPath}")
        sys.exit(1)
        
    print(f"Saved: {outputPath}")
    sys.exit(0)

