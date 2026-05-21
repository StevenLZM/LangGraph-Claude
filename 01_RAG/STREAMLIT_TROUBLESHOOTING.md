# Streamlit 启动报错排障记录

本文记录 `01_RAG` 执行 `streamlit run app.py` 时出现的报错原因、修复内容和验证方式。

## 一、问题现象

启动 Streamlit 时曾出现多类错误和警告：

- `DataDirLockedError: another process holds the lock`
- `ConnectionConfigException: Open local milvus failed`
- `Illegal uri: ./data/vectorstore/milvus.db`
- `transformers.models.zoedepth... ModuleNotFoundError: No module named 'torchvision'`
- 离线测试时 `tiktoken` 尝试下载 `cl100k_base.tiktoken` 失败

## 二、根因分析

### 1. Milvus Lite 被旧 Streamlit 进程占用

Milvus Lite 本地数据库同一时间只能被一个进程持锁。旧的 `streamlit run app.py` 进程未退出时，会占用：

```text
01_RAG/data/vectorstore/milvus.db/LOCK
```

再次启动应用会导致 Milvus 初始化失败。

### 2. `MILVUS_URI` 与 `pymilvus` 环境变量冲突

项目原来使用 `MILVUS_URI=./data/vectorstore/milvus.db` 表示本地 Milvus Lite 文件路径。但 `pymilvus` 也会读取同名环境变量，并期望它是远程 HTTP 地址，因此会把本地路径误判为非法 URI。

### 3. Streamlit 文件监听器触发 Transformers 可选依赖

安装 `watchdog` 后，Streamlit 默认文件监听器会扫描已导入模块。扫描 `transformers.models.zoedepth` 时触发 `torchvision` 导入；当前环境未安装 `torchvision`，于是输出 traceback。该问题不是 RAG 逻辑错误。

### 4. `tiktoken` 离线缓存缺失

分块模块使用 `cl100k_base` tokenizer。缓存不存在且网络不可用时，`tiktoken` 会尝试访问 OpenAI 公共 blob 地址，导致离线测试失败。

## 三、已完成修改

- `app.py`
  - 启动侧边栏时不再立即打开 Milvus。
  - 增加 Milvus Lite lock 检测和友好提示。
  - Chain 初始化失败时显示可读错误，而不是完整 traceback。

- `config.py`
  - 新增推荐配置项 `RAG_MILVUS_URI`。
  - 兼容旧 `MILVUS_URI`，读取后从环境变量移除，避免被 `pymilvus` 误读。

- `rag/vectorstore.py`
  - `get_collection_stats()` 增加失败兜底，向量库不可用时返回空统计和错误信息。

- `rag/chunker.py`
  - `tiktoken` 初始化失败时使用近似长度函数，保证离线测试可运行。

- `.streamlit/config.toml`
  - 增加：

```toml
[server]
fileWatcherType = "none"
```

该配置禁止 Streamlit 扫描第三方模块路径，避免 `transformers / zoedepth / torchvision` 噪音。

- `requirements.txt`
  - 增加 `watchdog`，减少 Streamlit 启动性能提示。

- `pytest.ini`
  - 注册 `slow` marker。
  - 过滤 PyMuPDF/SWIG 上游 deprecation warning。

## 四、推荐配置

`.env` 中使用：

```bash
RAG_MILVUS_URI=./data/vectorstore/milvus.db
```

不要再新增 `MILVUS_URI`。旧配置仍兼容，但不推荐继续使用。

## 五、常用排查命令

检查是否有 Streamlit 进程占用端口：

```bash
lsof -iTCP:8501 -sTCP:LISTEN
```

检查 Milvus Lite lock 是否被持有：

```bash
lsof 01_RAG/data/vectorstore/milvus.db/LOCK
```

如确认是旧的 01_RAG Streamlit 进程，可停止对应 PID：

```bash
kill <PID>
```

## 六、验证结果

已验证：

```bash
cd 01_RAG
pytest tests -q -k "not slow"
```

结果：

```text
93 passed, 1 deselected
```

Streamlit smoke test：

```bash
streamlit run app.py --server.headless true --server.port 8561 --browser.gatherUsageStats false
curl http://localhost:8561/_stcore/health
```

结果：

```text
ok
```

## 七、后续注意事项

- Milvus Lite 是本地单进程数据库，不适合多个 Streamlit 实例同时使用同一个 `milvus.db`。
- 如果需要多进程或多人并发访问，应切换到独立 Milvus 服务，而不是 Milvus Lite 文件。
- 关闭 `fileWatcherType` 后，修改代码不会自动热重载；开发时如需热重载，可以临时删除该配置，但可能重新触发 `transformers` 相关 watcher 警告。
