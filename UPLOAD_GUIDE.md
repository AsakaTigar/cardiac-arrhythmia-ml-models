# GitHub 和 HuggingFace 上传指南

## ✅ 准备完成清单

- [x] 交接包创建完成: `arrhythmia-prediction-handover-20251219.zip` (103MB)
- [x] 所有文档齐备 (README, LICENSE, etc.)
- [x] 模型文件已包含 (8个.joblib文件, 582MB)
- [x] 已配置.gitignore (保护患者数据)
- [x] Gradio演示应用已准备

---

## 📦 Package 1: GitHub Repository (Private)

### Step 1: 创建私有仓库

```bash
# 在GitHub上创建私有仓库
# https://github.com/new

Repository name: arrhythmia-prediction
Description: 心律失常预测模型 - 基于术前临床数据
Visibility: ☑️ Private (重要！)
Initialize: ☐ 不要添加README/LICENSE (我们已经有了)

# 点击 "Create repository"
```

### Step 2: 初始化并推送

```bash
cd /mnt/t2-6tb/Linpeikai/Shenxian_work_doc/handover_package

# 初始化Git
git init
git add .
git commit -m "Initial commit: Arrhythmia prediction models and documentation"

# 添加remote (替换YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/arrhythmia-prediction.git

# 推送
git branch -M main
git push -u origin main
```

### Step 3: 配置仓库设置

在GitHub仓库页面:
1. Settings → General → Features
   - ☑️ Issues
   - ☑️ Discussions (可选)
2. Settings → Security
   - 确认仓库是Private
3. Add topics: `machine-learning`, `healthcare`, `arrhythmia`, `xgboost`

---

## 🤗 Package 2: HuggingFace Space (Private)

### Step 1: 创建HuggingFace Account

1. 访问 https://huggingface.co/join
2. 注册账户
3. 生成Access Token:
   - Profile → Settings → Access Tokens
   - New token → Name: "arrhythmia-upload" → Role: Write
   - 复制token并保存

### Step 2: 安装HuggingFace CLI

```bash
# 在Aoduo环境中
conda activate Aoduo
pip install huggingface_hub

# 登录
huggingface-cli login
# 粘贴刚才复制的token
```

### Step 3: 创建私有Space

```bash
# 创建Gradio Space (私有)
huggingface-cli repo create arrhythmia-prediction \
    --type space \
    --space_sdk gradio \
    --private

# 克隆Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/arrhythmia-prediction
cd arrhythmia-prediction

# 复制文件
cp -r /mnt/t2-6tb/Linpeikai/Shenxian_work_doc/handover_package/* .

# 提交并推送
git add .
git commit -m "Initial upload: Arrhythmia prediction Gradio app"
git push
```

### Step 4: 配置Space

在HuggingFace Space页面:
1. Files and versions → Edit README
   - 添加Model Card信息
   - 说明是私有项目
2. Settings:
   - ☑️ Private (确认)
   - Sleep time: Never (保持运行)
   - Hardware: CPU Basic (免费)

---

## ⚠️ 重要注意事项

### ❌ 绝对不要上传

- `旧的/` 目录 (包含患者数据)
- 任何 `.xlsx`, `.xls` 原始数据文件
- 包含患者信息的CSV文件

### ✅ 可以上传

- 训练好的模型 (`.joblib`)
- 代码脚本 (`.py`)
- 文档 (`.md`)
- 结果统计 (`all_models_performance.csv` - 不含患者信息)
- Gradio应用

---

## 🔍 验证上传

### GitHub检查清单

- [ ] 仓库是**Private**
- [ ] README.md 正确显示
- [ ] LICENSE文件存在
- [ ] 所有代码文件可查看
- [ ] 无患者数据泄露

### HuggingFace检查清单

- [ ] Space是**Private**
- [ ] Gradio应用能正常启动
- [ ] 模型文件已上传
- [ ] README/Model Card完整
- [ ] 无患者数据泄露

---

## 📊 上传后确认

### 测试GitHub仓库

```bash
# 在新目录中克隆测试
git clone https://github.com/YOUR_USERNAME/arrhythmia-prediction.git test_repo
cd test_repo

# 检查文件
ls -lh
tree -L 2

# 尝试运行 (需要先准备数据)
python scripts/run_analysis.py --help
```

### 测试HuggingFace Space

1. 访问 Space URL
2. 等待应用启动 (~1-2分钟)
3. 输入测试数据:
   - LA: 45mm
   - INR: 1.2
   - 年龄: 65
   - 其他: 默认值
4. 点击"开始预测"
5. 验证输出格式正确

---

## 🔄 后续维护

### 更新代码

```bash
# GitHub
cd arrhythmia-prediction
# 修改文件...
git add .
git commit -m "Fix: updated model evaluation"
git push

# HuggingFace
cd arrhythmia-prediction  # (HF space目录)
# 修改文件...
git add .
git commit -m "Fix: improved Gradio interface"
git push
```

### 添加协作者

**GitHub**:
- Settings → Collaborators → Add people

**HuggingFace**:
- Settings → Members → Add member

---

## 📝 完成后的To-Do

1. 在本地保留原始数据备份 (不上传)
2. 记录仓库URL到安全位置
3. 与团队成员共享访问权限
4. 定期更新模型和文档
5. 监控Space运行状态

---

## 🆘 遇到问题?

### GitHub常见问题

**Q: Push被拒绝?**
```bash
# 确认remote URL
git remote -v

# 如果使用HTTPS需要token
# 使用SSH更方便
git remote set-url origin git@github.com:YOUR_USERNAME/arrhythmia-prediction.git
```

**Q: 文件太大?**
```bash
# Git不允许单文件>100MB
# RandomForest模型(569MB)需要Git LFS
git lfs install
git lfs track "*.joblib"
git lfs track "models/*"
git add .gitattributes
git commit -m "Add Git LFS for large model files"
```

### HuggingFace常见问题

**Q: Space启动失败?**
- 检查 `requirements.txt` 是否有冲突
- 查看 Space logs
- 确认 `app.py` 语法正确

**Q: 模型文件加载失败?**
- 确认文件路径正确 (`models/xxx.joblib`)
- 检查文件是否成功上传
- 验证文件大小是否正确

---

## ✅ 最终确认

完成上传后，请确认:

- [ ] GitHub仓库URL已记录
- [ ] HuggingFace Space URL已记录
- [ ] 两个仓库都是Private
- [ ] 无患者数据泄露
- [ ] README等文档完整
- [ ] 应用能正常运行
- [ ] 已与团队成员分享

**恭喜！项目交接准备完成！** 🎉

---

## 📎 参考链接

- GitHub Docs: https://docs.github.com/
- HuggingFace Spaces: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://www.gradio.app/docs
- Git LFS: https://git-lfs.com/
