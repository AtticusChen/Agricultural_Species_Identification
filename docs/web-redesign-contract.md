# 识别工作台 DOM / 行为契约

使用现有FastAPI API，无CDN、无前端构建。app.js为module，可import ./workspace-core.mjs。主编写HTML/CSS，JS代理不要改这两文件。

## 页面与导航

导航按钮属性data-view=workspace/history/guide；对应section id workspaceView/historyView/guideView，后两初始hidden。设置active类和aria-current，viewTitle更新中文导航标题。historyCount显示已完成及失败条数。用户明确无需手机布局，以电脑1440×1000、笔记本1280×800验收，无复制ID。

## 健康/上传/队列

healthIndicator元素data-state=loading/ready/offline；healthText；modelName（原始模型名）；connectionBanner初始hidden，reconnectButton。GET /api/v1/health status=ready；真实default_threshold/default_scan_mode记录，不硬编码模型已就绪。网络失败不崩溃，可重新连接。

fileInput multiple hidden，accept=image/jpeg,image/png,image/webp；chooseFiles按钮；dropZone是div role=button tabindex=0，点击/Enter/Space/拖放触发选择；addFiles按钮；uploadPanel在有文件时加has-files；queueCount显示N / 10；queueList是ul。

每个队列li.queue-item带data-status=pending/running/done/error和selected类。内部button.queue-select含img.queue-thumb、span.queue-info内strong.queue-name、span.queue-status；独立button.queue-remove含可访问名称移除文件名。文件名仅textContent，不拼innerHTML。缩略图复用objectURL，移除/清空销毁；最多10张，单20MB，总80MB，0字节拒绝；空mime可按支持扩展名fallback，重复name/size/lastModified不重复加入。

startButton开始pending/error项顺序识别（错误可重试），运行时禁新增/清空/删除/设置；pauseButton表示完成当前张后暂停，不能假称取消服务器已执行任务；clearButton清空前确认仅当已有结果，避免误丢记录。progressWrap初始hidden；progressBar内fill进度元素id progressFill，roleprogressbar在progressBar，更新aria-valuenow/min/max；progressText；notice是role=status aria-live。按钮icon由CSS/HTML已有内容保留，文字元素id startLabel。

## 示例

三个按钮data-sample=ground/aerial/other，实际GET /api/v1/examples返回 {examples:[{id,title,url,thumbnail}]}。按钮中img由HTML固定 /assets/media/thumb-ground.jpg等；点后fetch对应url获取Blob并new File加入队列，不使用假预测。接口失败有notice。文件名优先example的filename（可选），否则example-ground.jpg。例图均来自本项目且去EXIF。

## 设置

details高级设置包含threshold input range0..100 step0.1；thresholdValue只显示，实际默认不从滑块回传。resetThreshold恢复健康接口精确值；settingStatus默认“推荐设置”；custom改动才发threshold。扫描模式固定健康接口推荐default_scan_mode，整图多区域。仅用户改动阈值才带入FormData，否则省略；批次开始快照设置，运行中不变。

## 结果展示

resultEmpty初始显示；resultContent初始hidden；emptyTitle/emptyCopy可随未选择/处理/失败更新。resultFilename、resultBadge、resultTitle、resultExplanation。

原图/结果/对比按钮data-display=annotated/original/compare，父id viewTabs，默认annotated，未有结果禁annotated/compare。resultCanvas加display-original/display-annotated/display-compare类；originalFrame/img#originalImage，annotatedFrame/img#annotatedImage；通过hidden控制单/双图。图各caption HTML提供。待识别照片可以显示原图；失败显示原图+错误信息，不假定阴性。

statRegions（只称参考区域）、statScore（data.score缺失显示—，不得用最大区域得分冒充照片分数）、statTime、statThreshold、statModel。data.present是唯一照片阳性依据，可present=true而count=0。阳性文案“疑似发现薇甘菊”，负面“本图未检出薇甘菊”；说明粗区域非精确边界、得分非准确率、未检出不等于不存在。resultBadge data-tone=positive/negative/pending/error，resultContent可同样data-tone。

downloadImage下载真实annotated_image，校验为data:image/(jpeg|png|webp);base64，安全文件名；zoomButton打开dialog#imageDialog，dialogImage、dialogTitle、closeDialog；点击背景关闭、Esc原生支持。放大当前视图图片，对比模式默认放大标注图。

## 记录/导出

summaryTotal/summaryPositive/summaryNegative/summaryErrors计真实会话结果，不显示虚构运行统计。historyEmpty、historyTable（可hidden）、historyTableBody tbody；行包括文件名、结果、区域、得分、耗时、查看button.history-open[data-id]；data-filter=all/positive/negative/error按钮；historySearch。只显示done/error项。选中记录回workspace并展示真实对应结果。

exportCsv与exportCsvHistory按钮导出本会话done/error结果；0条禁用。CSV含BOM，字段文件名/结果/参考区域/判定得分/阈值/耗时ms/模型/时间，防公式注入=+-@，引号/换行正确转义。所有下载完成及时回收临时URL。会话记录刷新清空，不localStorage、不假持久化。

## 指南

data-view=guide导航切换即可；页面内button[data-view=workspace]返回上传，同一选择器委托。指南包含3步骤、相机照片/航拍建议、20MB/10张/80MB限制、误报漏检解释、连接失败重试和隐私说明。
