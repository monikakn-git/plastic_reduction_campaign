pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        CI_VENV = '.venv-ci'
    }

    stages {
        stage('Prepare Python') {
            steps {
                sh '''
                    python3.11 -m venv "$CI_VENV"
                    "$CI_VENV/bin/python" -m pip install --upgrade pip
                    "$CI_VENV/bin/pip" install -r requirements.txt flake8
                '''
            }
        }

        stage('Lint and syntax') {
            steps {
                sh '''
                    "$CI_VENV/bin/flake8" src/ --count --select=E9,F63,F7,F82 --show-source --statistics
                    "$CI_VENV/bin/flake8" src/ --count --exit-zero --max-line-length=120 --statistics
                    "$CI_VENV/bin/python" -m compileall -q src
                '''
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    "$CI_VENV/bin/python" -m pytest \
                        tests/test_deepchecks_compat.py \
                        tests/test_observability.py \
                        tests/test_registry.py \
                        tests/test_risk_quality_reporting.py \
                        tests/test_tracking.py -v
                '''
            }
        }

        stage('Validate dataset') {
            steps {
                sh '''
                    if [ -f artifacts/datasets/processed/data.yaml ]; then
                        "$CI_VENV/bin/python" src/data/validate_yolo_dataset.py --root artifacts/datasets/processed
                    else
                        echo 'Dataset files are absent from this checkout; skipping dataset validation.'
                    fi
                '''
            }
        }

        stage('Model quality gate') {
            steps {
                sh '''
                    if [ -f artifacts/checkpoints/yolo11n_best.pt ] && [ -f artifacts/datasets/processed/data.yaml ]; then
                        "$CI_VENV/bin/python" src/models/evaluate.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --split test --no-mlflow
                        "$CI_VENV/bin/python" src/models/risk_quality_checks.py --checkpoint yolo11n
                    else
                        echo 'Model checkpoint or dataset is absent from this checkout; skipping the model quality gate.'
                    fi
                '''
            }
        }

        stage('Build Docker images') {
            steps {
                sh '''
                    docker build --tag "ecopulse-api:${BUILD_NUMBER}" --file Dockerfile .
                    docker build --tag "ecopulse-mlflow:${BUILD_NUMBER}" --file Dockerfile.mlflow .
                '''
            }
        }

        stage('Deploy to Render') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'render-api-deploy-hook', variable: 'RENDER_API_DEPLOY_HOOK_URL'),
                    string(credentialsId: 'render-mlflow-deploy-hook', variable: 'RENDER_MLFLOW_DEPLOY_HOOK_URL')
                ]) {
                    sh '''
                        set +x
                        curl --fail --silent --show-error --request POST "$RENDER_MLFLOW_DEPLOY_HOOK_URL"
                        curl --fail --silent --show-error --request POST "$RENDER_API_DEPLOY_HOOK_URL"
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'rm -rf "$CI_VENV"'
        }
    }
}
