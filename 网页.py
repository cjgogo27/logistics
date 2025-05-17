<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>慧仓机器人-能源与空间优化方案</title>
    <link href="https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary-color: #2A9D8F;
            --secondary-color: #264653;
            --accent-color: #E9C46A;
        }

        body {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-family: 'Segoe UI', '微软雅黑', sans-serif;
        }

        .navbar {
            background: rgba(42, 157, 143, 0.95) !important;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }

        .hero-section {
            height: 80vh;
            background: linear-gradient(rgba(42, 157, 143, 0.8), rgba(38, 70, 83, 0.8)),
                        url('industrial-bg.jpg') center/cover;
            display: flex;
            align-items: center;
            color: white;
        }

        .section-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            margin-bottom: 2rem;
        }

        .section-card:hover {
            transform: translateY(-10px);
        }

        .data-visual {
            border-left: 4px solid var(--primary-color);
            padding-left: 1.5rem;
            margin: 2rem 0;
        }

        .interactive-chart {
            height: 500px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .accordion-button {
            background: var(--primary-color) !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg fixed-top">
        <div class="container">
            <a class="navbar-brand text-white fw-bold" href="#">慧仓优化方案</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link text-white" href="#energy">能源方案</a></li>
                    <li class="nav-item"><a class="nav-link text-white" href="#factory">厂房优化</a></li>
                    <li class="nav-item"><a class="nav-link text-white" href="#office">办公楼设计</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Banner -->
    <header class="hero-section">
        <div class="container text-center">
            <h1 class="display-4 fw-bold mb-4">智慧物流 绿色未来</h1>
            <p class="lead">慧仓机器人公司能源效率提升与空间优化综合解决方案</p>
        </div>
    </header>

    <!-- 主要内容 -->
    <main class="container py-5">
        <!-- 能源方案 -->
        <section id="energy" class="section-card p-5">
            <h2 class="mb-4 text-primary">绿色能源系统规划</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="interactive-chart" id="energy-comparison"></div>
                </div>
                <div class="col-md-6">
                    <div class="data-visual">
                        <h4>关键数据指标</h4>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between">
                                <span>光伏年发电量</span>
                                <span class="text-success">1,142,948 kWh</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>风电年发电量</span>
                                <span class="text-success">100,000 kWh</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>年度成本节约</span>
                                <span class="text-danger">¥208,758</span>
                            </li>
                        </ul>
                    </div>
                    <div class="accordion mt-4" id="energyAccordion">
                        <div class="accordion-item">
                            <h2 class="accordion-header">
                                <button class="accordion-button" type="button" data-bs-toggle="collapse"
                                    data-bs-target="#collapseOne">技术细节</button>
                            </h2>
                            <div id="collapseOne" class="accordion-collapse collapse show"
                                data-bs-parent="#energyAccordion">
                                <div class="accordion-body">
                                    <p>• 光伏系统：3163㎡屋顶面积，转换效率22%</p>
                                    <p>• 风电系统：2×40kW机组，年有效发电时长40%</p>
                                    <p>• 储能系统：90%充放电效率，1200kWh容量</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 厂房优化 -->
        <section id="factory" class="section-card p-5 mt-5">
            <h2 class="mb-4 text-primary">生产厂房空间优化</h2>
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="card h-100">
                        <img src="warehouse-layout.jpg" class="card-img-top" alt="布局示意图">
                        <div class="card-body">
                            <h5 class="card-title">智能仓储布局</h5>
                            <p class="card-text">采用立体货架+AGV系统，空间利用率提升40%</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="interactive-chart" id="factory-optimize"></div>
                    <div class="mt-3">
                        <button class="btn btn-outline-primary me-2" onclick="toggleLayout('3d')">3D布局</button>
                        <button class="btn btn-outline-primary" onclick="toggleLayout('2d')">平面视图</button>
                    </div>
                </div>
            </div>
        </section>

        <!-- 办公设计 -->
        <section id="office" class="section-card p-5 mt-5">
            <h2 class="mb-4 text-primary">绿色办公楼设计</h2>
            <div class="row">
                <div class="col-md-8">
                    <div class="interactive-chart" id="office-model"></div>
                </div>
                <div class="col-md-4">
                    <h5>多目标优化模型</h5>
                    <div class="progress my-3">
                        <div class="progress-bar bg-success" style="width: 85%">空间效率 85%</div>
                    </div>
                    <div class="progress my-3">
                        <div class="progress-bar bg-info" style="width: 78%">能源效率 78%</div>
                    </div>
                    <div class="progress my-3">
                        <div class="progress-bar bg-warning" style="width: 92%">沟通效率 92%</div>
                    </div>
                    <ul class="list-group">
                        <li class="list-group-item">
                            <i class="bi bi-people-fill me-2"></i>
                            人均办公面积 ≥8㎡
                        </li>
                        <li class="list-group-item">
                            <i class="bi bi-thermometer-sun me-2"></i>
                            温度舒适度指数 ≥0.8
                        </li>
                    </ul>
                </div>
            </div>
        </section>
    </main>

    <script src="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        // 能源对比图表
        const energyData = {
            labels: ['太阳能', '风能'],
            values: [1142948, 100000],
            colors: ['#2A9D8F', '#E9C46A']
        };

        const energyChart = {
            data: [{
                x: energyData.labels,
                y: energyData.values,
                type: 'bar',
                marker: {color: energyData.colors}
            }],
            layout: {
                title: '可再生能源发电量对比',
                height: 400
            }
        };
        Plotly.newPlot('energy-comparison', energyChart.data, energyChart.layout);

        // 动态切换布局函数
        function toggleLayout(type) {
            console.log(`切换为${type.toUpperCase()}布局`);
            // 此处可添加实际布局切换逻辑
        }

        // 页面滚动动画
        window.addEventListener('scroll', () => {
            const cards = document.querySelectorAll('.section-card');
            cards.forEach(card => {
                const cardTop = card.getBoundingClientRect().top;
                if(cardTop < window.innerHeight * 0.8) {
                    card.style.opacity = 1;
                    card.style.transform = 'translateY(0)';
                }
            });
        });
    </script>
</body>
</html>