# Bibliography notes (verified 2026-10-18)

Companion to `refs.bib` (36 entries). For each entry: key, full title, authors, venue + year, where it was verified, and what it contributes to our related work. **[DOUBT]** marks anything uncertain.

**How entries were verified.**
- Crossref DOI metadata (the publisher's own record), queried through api.crossref.org.
- The arXiv API and arXiv HTML full text.
- CVF open-access pages and PDFs.
- NeurIPS proceedings, plus a few author or institution pages.

DBLP and the ACM DL could not be reached: both returned bot challenges. Where DBLP would normally have been used, Crossref/IEEE/Springer metadata was used instead.

**Page ranges.** For papers in both IEEE Xplore and CVF open access, the bib uses the IEEE (DOI) range. The CVF range differs for some of them; those are listed below.

---

## 1. Learning-based dewarping (core)

### ma2018docunet
- **Title:** DocUNet: Document Image Unwarping via a Stacked U-Net
- **Authors:** Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, Dimitris Samaras
- **Venue:** CVPR 2018, pp. 4700-4709, DOI 10.1109/CVPR.2018.00494
- **Verified:** Crossref; CVF page https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html (the CVF BibTeX has no pages; the PDF's printed page 4704 is consistent with 4700-4709).
- **Summary:** The first deep-learning dewarping method. A stacked U-Net regresses a dense 2D mapping. It is trained on about 100K images synthesized by applying random 2D mesh perturbations (folds and curves) to flat document images. This is the closest precedent for our "synthetic 2D warps of flat pages" data strategy.
- **Benchmark:** Introduces the DocUNet benchmark of 65 paper documents with 2 phone photos each (130 images), plus flatbed-scan ground truth.
- **Metrics:** Introduces MS-SSIM/LD evaluation. Its own results are MS-SSIM 0.41 and LD 14.08.

### das2019dewarpnet
- **Title:** DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks
- **Authors:** Sagnik Das, Ke Ma, Zhixin Shu, Dimitris Samaras, Roy Shilkrot
- **Venue:** ICCV 2019, pp. 131-140, DOI 10.1109/ICCV.2019.00022
- **Verified:** Crossref; CVF PDF https://openaccess.thecvf.com/content_ICCV_2019/papers/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.pdf
- **Summary:** A shape network predicts a 3D coordinate map, and a texture-mapping network then regresses the backward map; a refinement network handles shading. The paper introduces **Doc3D**, about 100,000 rendered images built from real captured paper meshes, with 3D coordinates, UV, albedo, normals and backward-map annotations.
- **OCR protocol:** OCR is evaluated with PyTesseract v0.2.6 on text-heavy DocUNet images. The paper's text says "25 images from DocUNet"; later work (DocTr/DocGeoNet READMEs) describes this as 25 documents = 50 images ("Setting 2"). **[DOUBT: the original wording is ambiguous.]**

### feng2021doctr
- **Title:** DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction
- **Authors:** Hao Feng, Yuechen Wang, Wengang Zhou, Jiajun Deng, Houqiang Li
- **Venue:** ACM MM 2021, pp. 273-281, DOI 10.1145/3474085.3475388
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2110.12942 (comment: "accepted by ACM Multimedia 2021").
- **Summary:** A transformer-based geometric unwarping module (GeoTr) predicts a backward flow from background-removed input, and a separate illumination transformer (IllTr) follows it. It is trained on Doc3D.
- **OCR protocol:** Defines the 60-image (30-document) OCR subset now called "Setting 1".

### feng2022docgeonet
- **Title:** Geometric Representation Learning for Document Image Rectification
- **Authors:** Hao Feng, Wengang Zhou, Jiajun Deng, Yuechen Wang, Houqiang Li
- **Venue:** ECCV 2022, LNCS vol. 13697, pp. 475-492, DOI 10.1007/978-3-031-19836-6_27
- **Verified:** Crossref; Springer chapter page (LNCS volume); arXiv https://arxiv.org/abs/2210.08161
- **Summary:** Learns two explicit geometric representations: the 3D shape (coordinate map, L1 loss) and **text lines**. A text-line extractor is trained with per-pixel BCE on text-line masks of the distorted image. Its features are fused with the 3D-shape features in a transformer decoder that regresses the backward flow.
- **Text-line labels:** Generated automatically. Doc3D images are rectified, text-line boxes are found on the flat image, the horizontal lines are mapped back through the GT warp, and the result is curved-line masks. Text lines therefore enter as an **auxiliary segmentation target / feature**, not as a geometric loss on the predicted flow.
- **Datasets:** Introduces the DIR300 test set. Notes that the DocUNet images 127/128 (document 64) are rotated 180 degrees and reports results on the "corrected" benchmark.

### jiang2022rdgr
- **Title:** Revisiting Document Image Dewarping by Grid Regularization
- **Authors:** Xiangwei Jiang, Rujiao Long, Nan Xue, Zhibo Yang, Cong Yao, Gui-Song Xia
- **Venue:** CVPR 2022, IEEE pp. 4533-4542, DOI 10.1109/CVPR52688.2022.00450
- **Verified:** Crossref; CVF page and PDF https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html; arXiv 2203.16850
- **[NOTE] Pages:** The CVF BibTeX gives pp. 4543-4552. The bib uses the IEEE range.
- **Summary:** Detects document boundary points (by regression) and text-line pixels (UNet segmentation), then derives the vertical boundaries of text lines. It **solves a constrained optimization at inference time**: text lines should be horizontal and text-line boundaries vertical after dewarping, enforced through a grid-regularization term that yields the forward map. Text lines here are **hard geometric constraints in a per-image optimization**, not a training loss.
- **OCR protocol:** OCR is evaluated on 50 images with PyTesseract v0.3.8.

### zhang2022marior
- **Title:** Marior: Margin Removal and Iterative Content Rectification for Document Dewarping in the Wild
- **Authors:** Jiaxin Zhang, Canjie Luo, Lianwen Jin, Fengjun Guo, Kai Ding
- **Venue:** ACM MM 2022, pp. 2805-2815, DOI 10.1145/3503161.3548214
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2207.11515
- **Summary:** Two stages. A margin-removal module segments the document and removes margins. An iterative content-rectification module then predicts dense displacement flows, with the number of iterations chosen adaptively. It targets documents with large margins or no margins.

### feng2024doctrpp
- **Title:** Deep Unrestricted Document Image Rectification (DocTr++)
- **Authors:** Hao Feng, Shaokai Liu, Jiajun Deng, Wengang Zhou, Houqiang Li
- **Venue:** **IEEE Transactions on Multimedia**, vol. 26, pp. 6142-6154, 2024, DOI 10.1109/TMM.2023.3347094
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2304.08796 (v2 comment: "Accepted by TMM 2023").
- **[NOTE] Venue and year:** The journal version is TMM, not TPAMI or IJCV. Crossref dates the volume 2024; the authors' README BibTeX says 2023 (the online/accepted year).
- **Summary:** Extends DocTr with a hierarchical encoder-decoder for "unrestricted" inputs: full documents, partial crops, or no visible boundary. It introduces the UDIR test set.

### feng2025docscanner
- **Title:** DocScanner: Robust Document Image Rectification with Progressive Learning
- **Authors:** Hao Feng, Wengang Zhou, Jiajun Deng, Qi Tian, Houqiang Li
- **Venue:** IJCV vol. 133, no. 8, pp. 5343-5362, 2025, DOI 10.1007/s11263-025-02431-5
- **Verified:** Crossref; arXiv 2110.14968; GitHub README ("accepted by IJCV", 2025-03-24).
- **[NOTE] Pages:** The authors' README BibTeX says "pages 1--20" (online-first). The bib uses the final issue pages.
- **Summary:** Document localization followed by a recurrent (ConvGRU) progressive rectifier that refines a single flow estimate over iterations.
- **Relevant to our loss:** A **circle-consistency loss** maps each row and column of the GT flat image through the predicted backward flow and back through the GT forward flow, then penalizes deviation from a straight line. This is a **straightness constraint on every row and column of the page**, independent of content.
- **Metric:** Proposes the Line Distortion (Li-D) metric.

### ma2022paperedge
- **Title:** Learning From Documents in the Wild to Improve Document Unwarping (PaperEdge)
- **Authors:** Ke Ma, Sagnik Das, Zhixin Shu, Dimitris Samaras
- **Venue:** SIGGRAPH 2022 Conference Proceedings, DOI 10.1145/3528233.3530756
- **Verified:** Crossref; Stony Brook research portal https://researchconnect.stonybrook.edu/en/publications/learning-from-documents-in-the-wild-to-improve-document-unwarping/ (Article 34); GitHub cvlab-stonybrook/PaperEdge.
- **[NOTE] Pages:** The article number comes from the portal; Crossref shows "1-9".
- **Summary:** Collects about 5,000 real "Documents In the Wild" (DIW) photos with annotated paper boundaries. These are used for weakly supervised training alongside synthetic data, with a global edge-based network (Enet) and a texture-based refinement network (Tnet).
- **Metric:** Introduces the Aligned Distortion (AD) metric.

### xue2022fdrnet
- **Title:** Fourier Document Restoration for Robust Document Dewarping and Recognition (FDRNet)
- **Authors:** Chuhui Xue, Zichen Tian, Fangneng Zhan, Shijian Lu, Song Bai
- **Venue:** CVPR 2022, IEEE pp. 4563-4572, DOI 10.1109/CVPR52688.2022.00453
- **Verified:** Crossref; CVF page; arXiv 2203.09910
- **[NOTE] Pages:** The CVF BibTeX gives pp. 4573-4582.
- **Status:** Real paper.
- **Summary:** Works on high-frequency Fourier components, which carry structure but little appearance degradation. Dewarping uses a thin-plate-spline transform learned from lightly labeled real images, so no dense deformation GT is needed. It introduces the WarpDoc benchmark (about 1,020 camera documents).

### verhoeven2023uvdoc
- **Title:** UVDoc: Neural Grid-based Document Unwarping
- **Authors:** Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung
- **Venue:** SIGGRAPH Asia 2023 Conference Papers, pp. 1-11, DOI 10.1145/3610548.3618174
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2302.02887
- **Summary:** A small fully convolutional network jointly predicts a coarse 3D grid mesh and a 2D unwarping grid. It introduces the pseudo-photorealistic UVDoc training set (20K images) and the UVDoc benchmark.
- **Metric:** Proposes horizontal and vertical **line-straightness metrics**, which are useful for us as an evaluation precedent.
- **OCR protocol:** OCR is evaluated with Tesseract v4.0.0. Both images of DocUNet document 64 are **excluded** rather than rotated.

### yu2024docreal
- **Title:** DocReal: Robust Document Dewarping of Real-Life Images via Attention-Enhanced Control Point Prediction
- **Authors:** Fangchen Yu, Yina Xie, Lei Wu, Yafei Wen, Guozhi Wang, Shuai Ren, Xiaoxin Chen, Jianfeng Mao, Wenye Li
- **Venue:** WACV 2024, IEEE pp. 654-663, DOI 10.1109/WACV57701.2024.00072
- **Verified:** Crossref; CVF page https://openaccess.thecvf.com/content/WACV2024/html/Yu_DocReal_Robust_Document_Dewarping_of_Real-Life_Images_via_Attention-Enhanced_Control_WACV_2024_paper.html
- **[NOTE] Pages:** The CVF BibTeX gives pp. 665-674.
- **Summary:** Background removal plus an attention-enhanced control-point predictor. It introduces the DocReal benchmark (200 Chinese real-life images).
- **[NOTE] Naming:** "DocReal" and "UVDoc" are **two different papers**. The task list grouped them together; both are included.

### zhang2024docres
- **Title:** DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks
- **Authors:** Jiaxin Zhang, Dezhi Peng, Chongyu Liu, Peirong Zhang, Lianwen Jin
- **Venue:** CVPR 2024, pp. 15654-15664 (the same range in CVF and IEEE), DOI 10.1109/CVPR52733.2024.01482
- **Verified:** Crossref; CVF page; arXiv 2405.04408
- **Summary:** One model covers five restoration tasks: dewarping, deshadowing, appearance enhancement, deblurring and binarization. It is steered by a "Dynamic Task-Specific Prompt". For dewarping, the prompt is a document mask plus x/y coordinate maps.
- **[NOTE] Not layout-aware:** It was listed under layout-aware works in the request, but it uses no layout or text-line cues. Cite it only as a generalist baseline.

### liu2024drnet
- **Title:** Rethinking Supervision in Document Unwarping: A Self-Consistent Flow-Free Approach
- **Authors:** Shaokai Liu, Hao Feng, Wengang Zhou
- **Venue:** IEEE TCSVT vol. 34, no. 6, pp. 4817-4828, 2024, DOI 10.1109/TCSVT.2023.3336068
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2610.15445 (v1 posted 2026-10-11, comment "Accepted by TCSVT 2023").
- **[NOTE] Citation:** The arXiv ID is genuine but was posted long after journal publication. Cite the TCSVT version.
- **Summary:** "DRNet" is a coarse-to-fine rectifier supervised by the flat (undistorted) image instead of GT warping meshes, using low-level pixel alignment plus high-level semantic alignment and a geometric self-consistency term.
- **Results:** ForCenNet's table lists it on DocUNet at MS-SSIM 0.510, LD 7.42, ED 644.48, CER 0.164.

### wang2026aadd
- **Title:** Axis-Aligned Document Dewarping
- **Authors:** Chaoyun Wang, I-Chao Shen, Takeo Igarashi, Caigui Jiang
- **Venue:** AAAI 2026, Proc. AAAI vol. 40 (issue 12), pp. 9702-9710, DOI 10.1609/aaai.v40i12.37933
- **Verified:** Crossref; arXiv https://arxiv.org/abs/2507.15000 ("Accepted at AAAI 2026").
- **Status:** Real paper.
- **Summary:** Built on the UVDoc network.
- **Training loss:** An "axis-aligned geometric constraint" maps the predicted 2D unwarping grid into GT UV space and penalizes the variance of v along each row plus the variance of u along each column. The loss applies to **the whole grid, not to detected content**.
- **Inference:** Adds axis-alignment preprocessing.
- **Metric:** Proposes the Axis-Aligned Distortion (AAD) metric. All baselines were re-evaluated with Tesseract 5.4.0 at the original resolution.

---

## 2. Layout- and text-line-aware dewarping (positioning for our novelty claim)

**Summary for positioning.** Several recent works already use text lines or layout elements, and two of them (ForCenNet, MataDoc) use them in a **geometric training loss on the predicted flow**. That is the closest overlap with our layout-consistency loss.

Our safest defensible differences are these:
- (i) The loss is driven by **layout boxes (region-level annotations from PubLayNet)**. Prior work uses OCR text-line midlines, line segments, or segmentation masks.
- (ii) It penalizes **bending or distortion of whole text regions** rather than curvature or vertical consistency of individual line polylines.
- (iii) It uses free layout GT from born-digital pages, with no OCR or line detector needed at training time.

**Two cautions:**
- D2Dewarp (CVPR 2026) also renders **PubLayNet** pages and derives horizontal/vertical line masks from layout boundaries, but uses them as segmentation targets and features, not as a loss on the flow. Do not claim "first to use PubLayNet layout for dewarping".
- LA-DocFlatten is explicitly "layout-aware" (layout segmentation plus local UV maps). Do not claim "first layout-aware dewarping".

### li2024ladocflatten (LA-DocFlatten)
- **Title:** Layout-aware Single-image Document Flattening
- **Authors:** Pu Li, Weize Quan, Jianwei Guo, Dong-Ming Yan
- **Venue:** ACM Transactions on Graphics, vol. 43, no. 1, pp. 1-17, DOI 10.1145/3627818
- **Verified:** Crossref (with abstract); GitHub BunnySoCrazy/LA-DocFlatten.
- **[NOTE] Year:** Online 2023-11-02, issue dated 2024-02. Later papers cite it as "TOG'23". The article number could not be confirmed.
- **[NOTE] Authors:** The first author is Pu **Li**, not "Zhang" as given in the request.
- **How it uses layout:** Its insight is that most layout primitives are roughly rectangular, so unwarping each primitive locally is easier than unwarping the whole page.
  - A transformer segmentation module predicts the layout of the **distorted** input.
  - A regression module predicts global and local (per-primitive) UV maps.
  - A merging algorithm corrects the global prediction with the local ones.
- **Data and metric:** A synthetic dataset rendered with exact layout segmentation labels, plus a 200K deformed-mesh set, and a UV-based geometric correction metric.
- **Difference from ours:** Layout is used **architecturally, as divide-and-conquer local regressors**, not as a regularizing loss on a single dense flow.

### li2023ftdr (FTDR)
- **Title:** Foreground and Text-lines Aware Document Image Rectification
- **Authors:** Heng Li, Xiangping Wu, Qingcai Chen, Qianjin Xiang
- **Venue:** ICCV 2023, IEEE pp. 19517-19526, DOI 10.1109/ICCV51070.2023.01793
- **Verified:** Crossref; CVF page and PDF https://openaccess.thecvf.com/content/ICCV2023/html/Li_Foreground_and_Text-lines_Aware_Document_Image_Rectification_ICCV_2023_paper.html; GitHub xiaomore/Document-Image-Dewarping.
- **[NOTE] Pages:** The CVF and authors' BibTeX give pp. 19574-19583.
- **[DOUBT] Name:** The paper has no official short name. Later works call it "FTDR"; "FTRNet" is not used anywhere found.
- **How it uses text lines:**
  - Foreground masks and text-line masks are extracted from the distorted image. Text lines on Doc3D come from RDGR's detection method, because Doc3D lacks text-line labels.
  - The masks initialize foreground and text-line **queries** for cross-attention modules, which are fused in a transformer decoder.
  - The training loss is **only L1 on the grid**.
- **Difference from ours:** Text lines are an attention/feature cue, not a geometric loss.

### cai2025forcennet (ForCenNet)
- **Title:** ForCenNet: Foreground-Centric Network for Document Image Rectification
- **Authors:** Peng Cai, Qiang Li, Kaicheng Yang, Dong Guo, Jia Li, Nan Zhou, Xiang An, Ninghua Yang, Jiankang Deng
- **Venue:** ICCV 2025, pp. 15137-15146, DOI 10.1109/ICCV51701.2025.01405
- **Verified:** Crossref; CVF page; arXiv https://arxiv.org/abs/2507.19804 ("Accepted by ICCV25").
- **Status:** Real paper.
- **How it uses foreground and layout (closest to ours):**
  - Labels are generated on the **undistorted** image: an OCR engine gives text boxes, and their **midlines** are used as text-line representations; table lines come from an LSD-based detector; drawings are also extracted.
  - Labels are stored as masks and discrete points, and synthetic distortion fields (forward maps) are applied to flat images.
  - A foreground-centric mask mechanism supports a mask-guided transformer decoder.
  - A **curvature consistency loss** samples the line points through the predicted vs. GT backward map (bilinear), computes discrete curvature along each line, and penalizes the difference from GT curvature.
  - They report better straightening of text lines and table borders.
- **Difference from ours:**
  - Their loss supervises **curvature of 1D line elements** (OCR-box midlines, table lines). Ours uses **2D layout regions/boxes** and penalizes bending of the region as a whole.
  - Their labels come from OCR/LSD on flat images. Ours come from PubLayNet GT.
- **Positioning advice:** We must discuss ForCenNet explicitly and ideally compare to it.
- **[NOTE] Equation:** Eq. (6) in arXiv v1 is printed as a signed mean of (k_hat - k), likely a typo for an absolute or squared difference. Do not quote the formula verbatim.

### li2026d2dewarp (D2Dewarp)
- **Title:** D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping
- **Authors:** Heng Li, Xiangping Wu, Qingcai Chen (the FTDR group)
- **Venue:** CVPR 2026, pp. 34734-34744 (CVF)
- **Verified:** CVF page and PDF https://openaccess.thecvf.com/content/CVPR2026/html/Li_D2Dewarp_Dual_Dimensions_Geometric_Representation_Learning_Based_Document_Image_Dewarping_CVPR_2026_paper.html; arXiv 2507.08492 ("Accepted by CVPR 2026").
- **[NOTE] No DOI yet:** No IEEE/Crossref record was found, so the bib entry has no DOI.
- **How it uses layout:**
  - It builds **DocDewarpHV** (about 114K images, Blender-rendered). English pages come from the **PubLayNet** dev set; Chinese pages come from CDLA, CDDOD and M6Doc.
  - Layout annotations plus PaddleOCR text lines are merged into **horizontal-line masks** (top/bottom boundaries of the page, tables, figures, paragraphs and text lines) and **vertical-line masks** (left/right boundaries).
  - A dual-decoder UNet segments both line types on the distorted image (BCE/line losses), and an X/Y-coordinate fusion module combines them. The flow is supervised by a rectification loss.
- **Difference from ours:** Layout serves as an **auxiliary segmentation target and feature**, not as a constraint on the predicted flow.
- **Positioning advice:** This is the most similar *data* pipeline to ours (PubLayNet + layout-derived geometry), so cite it and state the difference clearly.

### dai2023matadoc (MataDoc) — arXiv only
- **Title:** MataDoc: Margin and Text Aware Document Dewarping for Arbitrary Boundary
- **Authors:** Beiya Dai, Xing Li, Qunyi Xie, Yulin Li, Xiameng Qin, Chengquan Zhang, Kun Yao, Junyu Han
- **Venue:** arXiv:2307.12571, 2023
- **Verified:** arXiv API and HTML full text.
- **[DOUBT] Peer review:** No peer-reviewed venue was found (Crossref search and web search both checked). Cite it as a preprint, or drop it if the venue requires peer-reviewed references only.
- **How it uses text lines:** A "text-aware regularization" (word-position consistency, loss L_T).
  - Text lines in the **rectified GT image** are obtained with an off-the-shelf OCR engine.
  - They are remapped to the distorted image with the predicted backward map, then mapped back with the GT forward map.
  - The loss constrains the **vertical coordinates of control points on the same text line to be equal**, which keeps text lines straight.
- **Margin regularization:** Also handles incomplete boundaries.
- **Difference from ours:** This is very close in spirit to a content-aware straightness loss. The difference is line-level y-consistency from OCR versus our region/box-level layout constraint.
- **Positioning advice:** Must be cited.

### liu2026textflow (TextFlow) — SSRN preprint
- **Title:** TextFlow: Textline-guided Generic Document Image Unwarping
- **Authors:** Shaokai Liu, Hao Feng, Yueding Zhu, Wuwen Liu, Hao Liu
- **Venue:** SSRN preprint, 2026, DOI 10.2139/ssrn.6842468
- **Verified:** Crossref record with abstract.
- **[DOUBT] Status:** An SSRN posting (Elsevier preprint server), likely under journal review; not peer-reviewed yet. Include only if preprints are acceptable.
- **How it uses text:**
  - Predicts an "unwarped text map" (text placement at each pixel of the rectified page) as an **auxiliary objective** alongside the flow.
  - Built on TextFlow1M, one million synthetic images with instance-level text annotations for both distorted and rectified pages.
- **Difference from ours:** Text/layout is an auxiliary prediction target, not a geometric constraint on the flow.

### Related works that use text lines / straightness without layout (already above)
- **DocGeoNet:** text-line segmentation as auxiliary representation.
- **RDGR:** text lines as constraints in an inference-time optimization.
- **DocScanner:** circle-consistency straightness of every GT row and column, content-agnostic.
- **AADD:** axis-aligned variance loss on the full predicted grid, content-agnostic.
- **UVDoc:** line-straightness *metrics* only.

### Seen but not added
DocTLNet (IJDAR 2024, cited in D2Dewarp's table) and SalmRec (arXiv 2505.06038, multi-task with text-line segmentation) were not verified in depth.

---

## 3. Classical (non-learning) dewarping

### brown2001document
- **Title:** Document Restoration Using 3D Shape: A General Deskewing Algorithm for Arbitrarily Warped Documents
- **Authors:** Michael S. Brown, W. Brent Seales
- **Venue:** ICCV 2001, vol. 2, pp. 367-374, DOI 10.1109/ICCV.2001.937649
- **Verified:** Crossref; mlanthology https://mlanthology.org/iccv/2001/brown2001iccv-document
- **[NOTE] Pages:** mlanthology lists 367-375; IEEE/Crossref 367-374 is used.
- **Summary:** Acquires 3D shape and texture of a warped document with a dedicated 3D digitization setup, then flattens it with a mass-spring particle simulation. This is the hardware/3D-reconstruction line of work.
- **Related:** The TPAMI 2004 follow-up exists ("Image restoration of arbitrarily warped documents", TPAMI 26(10):1295-1306, DOI 10.1109/TPAMI.2004.87) but was not added.

### tian2011rectification
- **Title:** Rectification and 3D Reconstruction of Curved Document Images
- **Authors:** Yuandong Tian, Srinivasa G. Narasimhan
- **Venue:** **CVPR 2011** (not ICCV 2011 as stated in the request), pp. 377-384, DOI 10.1109/CVPR.2011.5995540
- **Verified:** Crossref; Semantic Scholar record with abstract; the author PDF is hosted at CMU.
- **Summary:** From a single image, estimates a 2D distortion grid from **text-line structure and stroke statistics** (no binarization or character segmentation). It constrains the grid to be a projection of a 3D parallelogram mesh and recovers the 3D shape by a shape-from-texture SVD; shading is also removed. DocUNet compared against this method (MS-SSIM 0.13, LD 33.69).

### kim2015textline
- **Title:** Document Dewarping via Text-Line Based Optimization
- **Authors:** Beom Su Kim, Hyung Il Koo, Nam Ik Cho
- **Venue:** Pattern Recognition 48(11):3600-3614, 2015, DOI 10.1016/j.patcog.2015.04.026
- **Verified:** Crossref.
- **[DOUBT] Description:** The abstract could not be retrieved (publisher-elided). Going only by the title and the same group's later work, it optimizes a text-line-alignment cost. Read the paper before characterizing it in detail.

### kil2017robust
- **Title:** Robust Document Image Dewarping Method Using Text-Lines and Line Segments
- **Authors:** Taeho Kil, Wonkyo Seo, Hyung Il Koo, Nam Ik Cho
- **Venue:** ICDAR 2017, pp. 865-870, DOI 10.1109/ICDAR.2017.146
- **Verified:** Crossref; Semantic Scholar.
- **Summary:** Adds line segments to text-line cues, assuming well-rectified segments are straight and mostly horizontal or vertical. Outliers are removed iteratively, and page curvature, camera pose and focal length are estimated by cost minimization.
- **[DOUBT] Source of the description:** Taken from the first author's SNU PhD dissertation summary (core.ac.uk/works/86470992), not from the ICDAR abstract, which publishers elide.

---

## 4. Data, architecture, metrics, OCR, training

### zhong2019publaynet
- **Title:** PubLayNet: Largest Dataset Ever for Document Layout Analysis
- **Authors:** Xu Zhong, Jianbin Tang, Antonio Jimeno Yepes
- **Venue:** ICDAR 2019, pp. 1015-1022, DOI 10.1109/ICDAR.2019.00166
- **Verified:** Crossref; arXiv 1908.07836
- **Summary:** More than 360K document page images from PMC articles with automatically generated layout annotations (text, title, list, table, figure). This is our source of flat pages and layout boxes.

### ronneberger2015unet
- **Title:** U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors:** Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Venue:** MICCAI 2015, LNCS 9351, pp. 234-241, DOI 10.1007/978-3-319-24574-4_28
- **Verified:** Crossref; author page https://lmbweb.informatik.uni-freiburg.de/Publications/2015/RFB15a (LNCS volume); arXiv 1505.04597
- **Summary:** Our backbone architecture.

### liu2011siftflow
- **Title:** SIFT Flow: Dense Correspondence across Scenes and Its Applications
- **Authors:** Ce Liu, Jenny Yuen, Antonio Torralba
- **Venue:** IEEE TPAMI 33(5):978-994, **2011** (confirmed), DOI 10.1109/TPAMI.2010.147
- **Verified:** Crossref.
- **Summary:** Provides the dense correspondence used by the Local Distortion (LD) metric.

### wang2003msssim
- **Title:** Multiscale Structural Similarity for Image Quality Assessment
- **Authors:** Zhou Wang, Eero P. Simoncelli, Alan C. Bovik
- **Venue:** 37th Asilomar Conference on Signals, Systems & Computers, 2003, pp. 1398-1402, DOI 10.1109/ACSSC.2003.1292216
- **Verified:** Crossref.
- **[NOTE] Volume:** The Asilomar volume number was not confirmed and is omitted.
- **Summary:** The MS-SSIM metric. DocUNet-style evaluation uses weights 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 (as stated in DocTr++).

### smith2007tesseract
- **Title:** An Overview of the Tesseract OCR Engine
- **Authors:** Ray Smith
- **Venue:** ICDAR 2007, vol. 2, pp. 629-633, DOI 10.1109/ICDAR.2007.4376991
- **Verified:** Crossref.
- **Summary:** The OCR engine used for ED/CER.

### jaderberg2015stn
- **Title:** Spatial Transformer Networks
- **Authors:** Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
- **Venue:** NeurIPS 2015 (Advances in Neural Information Processing Systems 28)
- **Verified:** NeurIPS proceedings page https://papers.nips.cc/paper_files/paper/2015/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html; arXiv 1506.02025
- **[NOTE] Pages:** Not confirmed and omitted.
- **Summary:** Differentiable bilinear sampling, the basis of `grid_sample`-style warping with the predicted backward flow.

### loshchilov2019adamw
- **Title:** Decoupled Weight Decay Regularization
- **Authors:** Ilya Loshchilov, Frank Hutter
- **Venue:** ICLR 2019
- **Verified:** arXiv 1711.05101 ("Published as a conference paper at ICLR 2019"); iclr.cc/virtual/2019/poster/935; mlanthology.
- **Summary:** AdamW.

### loshchilov2017sgdr
- **Title:** SGDR: Stochastic Gradient Descent with Warm Restarts
- **Authors:** Ilya Loshchilov, Frank Hutter
- **Venue:** ICLR 2017
- **Verified:** arXiv 1608.03983 ("ICLR 2017 conference paper"); mlanthology.
- **Summary:** Cosine annealing schedule.

### micikevicius2018mixed
- **Title:** Mixed Precision Training
- **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu
- **Venue:** ICLR 2018
- **Verified:** arXiv 1710.03740 ("Published as a conference paper at ICLR 2018"); mlanthology.
- **Summary:** FP16 training with loss scaling.

### wilcoxon1945individual
- **Title:** Individual Comparisons by Ranking Methods
- **Authors:** Frank Wilcoxon
- **Venue:** Biometrics Bulletin 1(6):80-83, 1945, DOI 10.2307/3001968
- **Verified:** Crossref (volume, issue, first page 80); MathWorld references (pp. 80-83).
- **Summary:** The signed-rank test used for paired per-image comparisons.

### efron1993bootstrap
- **Title:** An Introduction to the Bootstrap
- **Authors:** Bradley Efron, Robert J. Tibshirani
- **Venue:** Chapman & Hall, 1993 (book), ISBN 0-412-04231-2, DOI 10.1007/978-1-4899-4541-9
- **Verified:** Crossref. The DOI record names "Springer US, Boston, MA" as the current holder of the original Chapman & Hall edition; a 1994 CRC reprint also exists.
- **Summary:** Bootstrap confidence intervals.

---

## 5. Dropped or corrected items

**Nothing requested was fabricated; every requested paper exists.** Corrections:

- **DocTr++:** The journal is **IEEE TMM** (2024, vol. 26), not TPAMI/IJCV.
- **DocScanner:** Confirmed as IJCV 2025, 133(8).
- **Layout-aware Single-image Document Flattening:** Authors are **Li, Quan, Guo, Yan** (not Zhang); ACM TOG 43(1), 2024 (online 2023).
- **"Foreground and Text-lines Aware…":** Li et al., ICCV 2023, is real. The short name used by others is "FTDR", not "FTRNet".
- **Tian & Narasimhan:** **CVPR 2011**, not ICCV 2011.
- **arXiv 2610.15445:** Real, but it is a late arXiv posting of a **TCSVT 2024** paper (DRNet). Cite TCSVT.
- **"DocReal / UVDoc":** Two separate papers; both added.
- **DocRes:** Not a layout-aware method (see note above).
- **Added, not requested:** D2Dewarp (CVPR 2026), MataDoc (arXiv), TextFlow (SSRN preprint), Kim et al. 2015 (PR). These are directly relevant to the novelty claim.

---

## 6. DocUNet benchmark results table (re-confirmed from sources)

### Setting definitions
- **Setting 1 (S1):** 60 images (30 documents), the DocTr OCR subset (`ocr_img.txt` in the DocTr/DocGeoNet/DocScanner repos).
- **Setting 2 (S2):** 50 images (25 documents), the DewarpNet OCR subset.
- **Feng-group re-evaluation:** DocGeoNet, DocScanner and DocTr++ use Tesseract 5.0.1 (Windows build 5.0.1.20220118) with pytesseract 0.3.8, MATLAB 2019a for MS-SSIM/LD, and the **corrected** benchmark (DocUNet images 127/128, i.e. document 64, rotated by 180 degrees to match GT).
- **Notation:** DocGeoNet and DocScanner print ED and CER as "S2/S1"; the paper text says "we select 50 and 60 images respectively". This matches the GitHub READMEs, where Setting 1 gives DocTr ED 464.83 and DocGeoNet ED 379.00.

### Sources
- **[A]** DocTr++ (arXiv 2304.08796v2) Table I. Only one ED/CER column: **S1, 60 images** ("following DocTr"), Tesseract 5.0.1, corrected benchmark (stated explicitly). MS-SSIM is given to 2 decimals. The table was read from the arXiv HTML; the TMM camera-ready was not checked.
- **[B]** DocGeoNet (arXiv 2210.08161) Table 1, corrected benchmark, S2/S1.
- **[C]** DocScanner (arXiv 2110.14968v3) Table 2 plus README, corrected benchmark, S2/S1.
- **[R]** GitHub READMEs of DocTr / DocGeoNet / DocScanner (fh2019ustc), which give S1 and S2 explicitly.
- **[U]** UVDoc (arXiv 2302.02887) Table 2. Tesseract **4.0.0**, document 64 **excluded**, OCR subset not stated, ED rounded to an integer.
- **[F]** ForCenNet (arXiv 2507.19804) Table 1. One ED/CER column; the values match S2 for most rows.
- **[X]** AADD (arXiv 2507.15000) Table 1. All methods re-run with Tesseract **5.4.0** at original resolution, "25 rich-text images" (presumably the S2 documents), corrected benchmark.
- **[D]** D2Dewarp (CVPR 2026) Table 2, S2/S1, Tesseract 5.0.1.
- **[O]** The method's own original paper.

### Main table: corrected DocUNet benchmark, Feng-group protocol unless noted

| Method | MS-SSIM ↑ | LD ↓ | ED S1 (60) ↓ | CER S1 ↓ | ED S2 (50) ↓ | CER S2 ↓ | Source(s) |
|---|---|---|---|---|---|---|---|
| Distorted | 0.2459 | 20.51 | 1552.22 | 0.5089 | 2111.56 | 0.5352 | B, C; A gives 0.25 / 20.51 / 1552.22 / 0.5089 |
| DocUNet | 0.4103 | 14.19 | 1259.83 | 0.3966 | 1933.66 | 0.4632 | B, C; A agrees (S1) |
| DewarpNet | 0.4735 | 8.39 | 525.45 | 0.2102 | 885.90 | 0.2373 | B, C; A agrees (S1); F agrees (S2) |
| DocTr (GeoTr) | 0.5105 | 7.76 | 464.83 | 0.1746 | 724.84 | 0.1832 | B, C, R; A agrees (0.51 / 7.76 / 464.83 / 0.1746) |
| DocGeoNet | 0.5040 | 7.71 | 379.00 | 0.1509 | 713.94 | 0.1821 | B, C, R; A agrees |
| RDGR | 0.4968 | 8.51 | 420.25 | 0.1559 | 729.52 | 0.1717 | B, C; A agrees (0.50 / 8.51 / 420.25 / 0.1559) |
| Marior | 0.4780 | 7.44 | 593.80 | 0.2136 | 776.22 | 0.1928 | C; A agrees on S1 (0.48 / 7.44 / 593.80 / 0.2136) |
| DocTr++ | 0.51 | 7.52 | 447.47 | 0.1695 | n/a | n/a | A only (S1 only) |
| FDRNet | 0.5420 | 8.21 | 514.90 | 0.1846 | 829.78 | 0.2068 | B; F has S2 829.78 / 0.206 |
| PaperEdge | 0.4724 | 7.99 | 375.60 | 0.1541 | 777.76 | 0.2014 | C; D agrees (777.76/375.60) |
| DocScanner-B | 0.5134 | 7.62 | 434.11 | 0.1652 | 671.48 | 0.1789 | C (Table 4 prints 0.1788), R |
| DocScanner-L | 0.5178 | 7.45 | 390.43 | 0.1486 | 632.34 | 0.1648 | C, R; D prints 0.52 / 7.45 / 390.40 / 0.1490 / 632.30 / 0.1650 |
| UVDoc | 0.55 | 6.79 | 493.13 | 0.1611 | 797.92 | 0.1975 | D (re-run by D2Dewarp authors) |
| FTDR | 0.4978 | 8.43 | 450.92 | 0.1679 | 697.52 | 0.1705 | O gives S2 only (Tesseract 5.0.1, corrected); S1 from D |
| LA-DocFlatten | 0.53 | 6.72 | 391.90 | 0.1530 | 695.00 | 0.1750 | D; F: 0.526 / 6.72, no OCR |
| D2Dewarp | 0.50 | 7.71 | 351.08 | 0.1338 | 656.30 | 0.1543 | D (own) |

### Other self-reported or differently evaluated numbers (not directly comparable)

| Method | MS-SSIM | LD | ED | CER | Protocol / source |
|---|---|---|---|---|---|
| DocUNet (orig.) | 0.41 | 14.08 | – | – | O; uncorrected benchmark |
| DewarpNet (orig.) | 0.4692 (ref: 0.4735) | 8.98 (ref: 8.95) | 1288.60 (ref: 1114.40) | 0.3136 (ref: 0.2692) | O; PyTesseract v0.2.6, "25 images" |
| DocTr (orig., MM'21) | 0.4970 (GeoTr+IllTr) | 8.38 (GeoTr) | 935.2 (GeoTr) / 576.4 (GeoTr+IllTr) | 0.31 / 0.20 | O; Tesseract **v3.02.02**, uncorrected benchmark |
| RDGR (orig.) | 0.4922 | 9.36 | 896.48 | 0.2068 | O; 50 images, PyTesseract v0.3.8, uncorrected |
| UVDoc (orig.) | 0.544 | 6.83 | 707 | 0.172 | U; Tesseract 4.0.0, document 64 excluded |
| UVDoc (AADD re-run) | 0.545 | 6.827 | 754.2 | 0.193 | X |
| ForCenNet | 0.582 | 4.82 | 571.40 | 0.136 | F (own); OCR subset not stated explicitly |
| AADD (full) | 0.543 | 6.249 | 603.1 | 0.150 | X (own) |
| DRNet (TCSVT) | 0.510 | 7.42 | 644.48 | 0.164 | F (listed as weakly supervised) |

### Inconsistencies between sources
1. **DocTr: the original MM'21 numbers differ from all later Feng-group numbers.** For example, MM'21 reports ED 576.4 / CER 0.20 for GeoTr+IllTr and LD 8.38 for GeoTr; the later numbers are ED 464.83 (S1) / LD 7.76. The causes are a different Tesseract (3.02.02 vs 5.0.1), the uncorrected vs. corrected benchmark, and the inclusion of illumination correction. The DocTr README says the MM-paper numbers used the "mistaken" samples.
2. **DocUNet LD:** 14.08 in the original paper and in DocTr's table vs. 14.19 in DocGeoNet/DocScanner/DocTr++ (corrected benchmark re-evaluation).
3. **DewarpNet:** The original reports MS-SSIM 0.4692 (0.4735 with refinement) and LD 8.98. Later tables use 0.4735 / 8.39, i.e. the refinement-network MS-SSIM with a re-computed LD. UVDoc reports 0.472 / 8.38, and AADD 0.474 / 8.362.
4. **RDGR:** The original paper (0.4922 / 9.36 / 896.48 / 0.2068) and the Feng-group re-evaluation (0.4968 / 8.51 / 729.52 / 0.1717) differ substantially. FTDR prints RDGR MS-SSIM as 0.4950, not 0.4968. RDGR's own table labels as "DocUNet" the numbers that DewarpNet reported for "DocUNet on Doc3D".
5. **Marior, PaperEdge and DDCP in ForCenNet:** ForCenNet's single ED/CER column matches S2 for DewarpNet, DocTr, DocGeoNet, RDGR, FDRNet and PWUNet. It does **not** match S2 for these three:
   - Marior: 823.80 / 0.205 vs. S2 776.22 / 0.1928.
   - PaperEdge: 825.48 / 0.211, and MS-SSIM 0.470 / LD 8.49, vs. 777.76 / 0.2014 / 0.4724 / 7.99.
   - DDCP: 1411.38 / 0.357 vs. 1442.84 / 0.3633. (FTDR's table also gives DDCP 1411.38 / 0.3573 and MS-SSIM 0.4726 / LD 8.97, vs. DocGeoNet's 0.4729 / 8.99.)

   These look like independent re-runs; **do not mix these columns.**
6. **FDRNet:** The S2 CER is 0.2068 in DocGeoNet and FTDR, and 0.206 in ForCenNet. UVDoc's re-evaluation gives 0.543 / 8.08 / 878 / 0.214.
7. **DocScanner-B:** S2 CER is 0.1789 in Table 2 and the README, but 0.1788 in Table 4 of the same paper.
8. **Document 64 handling:**
   - DocGeoNet, DocScanner, DocTr++ (stated in its Sec. V-A), FTDR and AADD **rotate** images 127/128.
   - D2Dewarp's text does not say how it handles them, but its baseline numbers match the corrected-benchmark values.
   - UVDoc **excludes** document 64 (128 images).
   - Most pre-2022 papers used the mistaken images.
9. **The DewarpNet OCR subset wording:** The original text says "25 images"; later works describe the DewarpNet subset as 25 documents = 50 images (Setting 2). AADD says "25 rich-text images".
10. **LA-DocFlatten OCR:** Not reported by ForCenNet. D2Dewarp gives 695.00 / 391.90 (S2/S1). AADD's re-run (with downscaling of LADoc outputs) gives 689.8 / 0.180.

### Recommendation for our paper
Report both S1 and S2 using the corrected benchmark, Tesseract 5.0.1 and pytesseract 0.3.8 (the DocGeoNet/DocScanner protocol). For baselines, copy the B/C/R rows above. Do not mix in UVDoc-, AADD- or ForCenNet-protocol numbers without labelling them. State the Tesseract version, because ED/CER are very sensitive to it; UVDoc reports large run-to-run OCR variance (CER ±0.012-0.018 over 10 trainings).
