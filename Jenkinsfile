pipeline {
  agent { docker { image 'python:3.7' } }
  stages {
    stage('build') {
      steps {
        sh 'pip install -r requirements.txt'
      }
    }
    stage('test') {
      steps {
        sh 'test_selective_search.py'
      }   
    }
  }
}
