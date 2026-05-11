# Acoustics Agent Studio (acoustics-agent-app)

[![使用说明](https://img.shields.io/badge/📖_文档-使用说明-green) ](docs/使用说明.md) [![User Guide](https://img.shields.io/badge/📖_Docs-User_Guide-blue) ](docs/User_Guide.md) [![English README](https://img.shields.io/badge/🌐_README-English-red) ](README.md)

`acoustics-agent-app` 是一个基于 `acoustics-agent` (AI-原生水声仿真框架) 构建的现代化、交互式的 Web 应用与决策平台。

![Acoustics Studio Munk Ray Paths](docs/munk_ray_paths.png)

它为海洋学家、水声工程师和科研人员提供了一个直观且视觉效果出色的图形用户界面 (GUI)，使用户能够设计物理波导、执行射线追踪和简正波仿真、分析多径到达特性，并绘制相干声场干涉图——这一切完全依赖标准的 Python 科学计算库。

---

## ✨ 核心特性

- **📊 实时 SSP 设计器**: 交互式声速剖面 (SSP) 编辑器。在数据表中调整深度和声速，右侧的 Plotly 折线图会实时更新。
- **🚢 波导与环境预设**: 提供基准海洋环境场景的“一键加载”模板：
  - **Munk 深海声道** (Deep-Ocean Sound Channel)
  - **Pekeris 浅海波导** (Shallow-Water Waveguide，含沙质海底反射)
  - **表面声道** (Surface Duct，混合层捕获)
- **📐 完整的几何控制**: 通过滑动条和数值输入框，全方位控制声源深度、水深、发射频率、掠射角、声线密度、海底属性等。
- **📈 交互式射线轨迹**: 采用 Plotly 和 Matplotlib 双重渲染。轻松平移、缩放以及悬停查看仿真声线细节。
- **🌈 相干传播损失热力图**: 采用高保真的 **高斯束求和 (Gaussian Beam Summation)** 算法，渲染呈现相干声场的干涉条纹。
- **🌋 非相干 TL 热力图**: 绘制基于射线密度的声能量衰减传播图。
- **🔔 多径冲激响应分析**: 自定义接收器的距离和深度，计算精确的声学多径到达路径，包括时延、振幅以及边界反射次数。
- **🤖 规则 AI 助手 (Co-pilot)**: 允许用户输入自然语言指令（例如：*"Set source depth to 150m, change frequency to 300Hz"*），瞬间自动解析并更新面板参数。

---

## 🚀 快速启动

1. **安装环境依赖**
   在 `acoustics-agent-app` 目录下，确保您拥有包含 Streamlit、Plotly 等模块的 Python 环境：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行 Studio**
   使用 Streamlit 启动应用服务器：
   ```bash
   streamlit run app.py
   ```

3. **浏览器访问**
   应用启动后，打开浏览器并访问 `http://localhost:8501` 即可体验。

有关各项功能模块的具体操作与图文引导，请参考 [使用说明.md](docs/使用说明.md)。
