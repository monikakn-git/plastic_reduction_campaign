pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        CI_VENV = '.venv-ci'
        PYTHONUTF8 = '1'
        PYTHONDONTWRITEBYTECODE = '1'
    }

    stages {
        stage('Prepare Python') {
            steps {
                bat '''
                    @echo off
                    if exist "%CI_VENV%" rmdir /s /q "%CI_VENV%"
                    where.exe python
                    if errorlevel 1 exit /b 1
                    python --version
                    if errorlevel 1 exit /b 1
                    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Python 3.11 or newer is required')"
                    if errorlevel 1 exit /b 1
                    python -m venv "%CI_VENV%"
                    if errorlevel 1 exit /b 1
                    "%CI_VENV%\\Scripts\\python.exe" -m pip install --upgrade pip
                    if errorlevel 1 exit /b 1
                    "%CI_VENV%\\Scripts\\python.exe" -m pip install -r requirements.txt flake8
                '''
            }
        }

        stage('Lint and syntax') {
            steps {
                bat '''
                    @echo off
                    "%CI_VENV%\\Scripts\\flake8.exe" src/ --count --select=E9,F63,F7,F82 --show-source --statistics
                    if errorlevel 1 exit /b 1
                    "%CI_VENV%\\Scripts\\flake8.exe" src/ --count --exit-zero --max-line-length=120 --statistics
                    if errorlevel 1 exit /b 1
                    "%CI_VENV%\\Scripts\\python.exe" -m compileall -q src
                '''
            }
        }

        stage('Unit tests') {
            steps {
                bat '''
                    @echo off
                    "%CI_VENV%\\Scripts\\python.exe" -m pytest tests\\test_deepchecks_compat.py tests\\test_observability.py tests\\test_registry.py tests\\test_risk_quality_reporting.py tests\\test_tracking.py -v
                '''
            }
        }

        stage('Validate dataset') {
            steps {
                bat '''
                    @echo off
                    if exist artifacts\\datasets\\processed\\data.yaml (
                        "%CI_VENV%\\Scripts\\python.exe" src\\data\\validate_yolo_dataset.py --root artifacts\\datasets\\processed
                        if errorlevel 1 exit /b 1
                    ) else (
                        echo Dataset files are absent from this checkout; skipping dataset validation.
                    )
                '''
            }
        }

        stage('Model quality gate') {
            steps {
                bat '''
                    @echo off
                    if exist artifacts\\checkpoints\\yolo11n_best.pt (
                        if exist artifacts\\datasets\\processed\\data.yaml (
                            "%CI_VENV%\\Scripts\\python.exe" -m pip install -r requirements-quality.txt
                            if errorlevel 1 exit /b 1
                            "%CI_VENV%\\Scripts\\python.exe" src\\models\\evaluate.py --checkpoint artifacts\\checkpoints\\yolo11n_best.pt --split test --no-mlflow
                            if errorlevel 1 exit /b 1
                            "%CI_VENV%\\Scripts\\python.exe" src\\models\\risk_quality_checks.py --checkpoint yolo11n
                            if errorlevel 1 exit /b 1
                        ) else (
                            echo Dataset is absent from this checkout; skipping the model quality gate.
                        )
                    ) else (
                        echo Model checkpoint is absent from this checkout; skipping the model quality gate.
                    )
                '''
            }
        }

        stage('Build Docker images') {
            steps {
                bat '''
                    @echo off
                    where.exe docker
                    if errorlevel 1 exit /b 1
                    docker version
                    if errorlevel 1 exit /b 1
                    docker build --tag "ecopulse-api:%BUILD_NUMBER%" --file Dockerfile .
                    if errorlevel 1 exit /b 1
                    docker build --tag "ecopulse-mlflow:%BUILD_NUMBER%" --file Dockerfile.mlflow .
                '''
            }
        }

        stage('Deploy to Render') {
            when {
                branch 'master'
            }
            steps {
                withCredentials([
                    string(credentialsId: 'render-api-deploy-hook', variable: 'RENDER_API_DEPLOY_HOOK_URL'),
                    string(credentialsId: 'render-mlflow-deploy-hook', variable: 'RENDER_MLFLOW_DEPLOY_HOOK_URL')
                ]) {
                    bat '''
                        @echo off
                        curl.exe --fail --silent --show-error --request POST "%RENDER_MLFLOW_DEPLOY_HOOK_URL%"
                        if errorlevel 1 exit /b 1
                        curl.exe --fail --silent --show-error --request POST "%RENDER_API_DEPLOY_HOOK_URL%"
                    '''
                }
            }
        }
    }

    post {
        always {
            bat '@echo off\r\nif exist "%CI_VENV%" rmdir /s /q "%CI_VENV%"'
        }
    }
}
