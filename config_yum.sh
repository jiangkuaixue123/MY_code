# 清理所有仓库配置
rm -f /etc/yum.repos.d/*.repo

# 使用阿里云镜像（通常在国内网络环境更稳定）
cat >/etc/yum.repos.d/openeuler.repo <<'EOF'
[OS]
name=openEuler-24.03-LTS - OS
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/OS/$basearch/
enabled=1
gpgcheck=0
sslverify=0

[everything]
name=openEuler-24.03-LTS - everything
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/everything/$basearch/
enabled=1
gpgcheck=0
sslverify=0

[EPOL]
name=openEuler-24.03-LTS - EPOL
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/EPOL/main/$basearch/
enabled=1
gpgcheck=0
sslverify=0
EOF

# 清理缓存
yum clean all
yum makecache