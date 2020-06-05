pipeline {
    agent any 
    stages {
        stage('Static Analysis') {
            steps {
                echo 'Run the  analysis to the code' 
                echo "${env.BRANCH_NAME}"
            }
        }
        stage('Trial stage') {
            when {
                expression {
                    env.BRANCH_NAME == 'master'
                }
            }
            steps {
                echo 'Finale' 
            }
        }
        stage('Ngrock stage') {
            steps {
                echo 'Run the trial for ngrock' 
            }
        }
        stage('Compile') {
            steps {
                echo 'Compile the source code' 
            }
        }
        stage('Security Check') {
            steps {
                echo 'Run the security check against the application' 
            }
        }
        stage('Run Unit Tests') {
            steps {
                echo 'Run unit tests from the source code' 
            }
        }
        stage('Run Integration Tests') {
            steps {
                echo 'Run only crucial integration tests from the source code' 
            }
        }
        stage('Publish Artifacts') {
            steps {
                echo 'Save the assemblies generated from the compilation' 
            }
        }
    }
}
