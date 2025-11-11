'use client';

import { useState } from 'react';
import styles from './DeploymentOptions.module.css';

interface DeploymentOptionsProps {
  modelId: string;
  onDeploy: (type: 'local' | 'e2b' | 'aws', awsKeys?: { accessKey: string; secretKey: string; region: string }) => void;
}

export function DeploymentOptions({ modelId, onDeploy }: DeploymentOptionsProps) {
  const [deploymentType, setDeploymentType] = useState<'local' | 'e2b' | 'aws'>('e2b');
  const [showAwsForm, setShowAwsForm] = useState(false);
  const [awsAccessKey, setAwsAccessKey] = useState('');
  const [awsSecretKey, setAwsSecretKey] = useState('');
  const [awsRegion, setAwsRegion] = useState('us-east-1');

  const handleDeploy = () => {
    if (deploymentType === 'aws' && (!awsAccessKey || !awsSecretKey)) {
      alert('Please provide AWS credentials');
      return;
    }

    if (deploymentType === 'aws') {
      onDeploy('aws', { accessKey: awsAccessKey, secretKey: awsSecretKey, region: awsRegion });
    } else {
      onDeploy(deploymentType);
    }
  };

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Deploy Model</h3>
      
      <div className={styles.options}>
        <label className={styles.option}>
          <input
            type="radio"
            name="deployment"
            value="local"
            checked={deploymentType === 'local'}
            onChange={(e) => {
              setDeploymentType('local');
              setShowAwsForm(false);
            }}
          />
          <div className={styles.optionContent}>
            <div className={styles.optionTitle}>Run Locally</div>
            <div className={styles.optionDesc}>Download model and run on your machine</div>
          </div>
        </label>

        <label className={styles.option}>
          <input
            type="radio"
            name="deployment"
            value="e2b"
            checked={deploymentType === 'e2b'}
            onChange={(e) => {
              setDeploymentType('e2b');
              setShowAwsForm(false);
            }}
          />
          <div className={styles.optionContent}>
            <div className={styles.optionTitle}>E2B Deployment (Basic)</div>
            <div className={styles.optionDesc}>Deploy on E2B sandbox (very slow, free tier)</div>
          </div>
        </label>

        <label className={styles.option}>
          <input
            type="radio"
            name="deployment"
            value="aws"
            checked={deploymentType === 'aws'}
            onChange={(e) => {
              setDeploymentType('aws');
              setShowAwsForm(true);
            }}
          />
          <div className={styles.optionContent}>
            <div className={styles.optionTitle}>AWS Deployment</div>
            <div className={styles.optionDesc}>Deploy on AWS with your credentials</div>
          </div>
        </label>
      </div>

      {showAwsForm && (
        <div className={styles.awsForm}>
          <div className={styles.field}>
            <label>AWS Access Key ID</label>
            <input
              type="text"
              value={awsAccessKey}
              onChange={(e) => setAwsAccessKey(e.target.value)}
              placeholder="AKIAIOSFODNN7EXAMPLE"
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label>AWS Secret Access Key</label>
            <input
              type="password"
              value={awsSecretKey}
              onChange={(e) => setAwsSecretKey(e.target.value)}
              placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label>Region</label>
            <select
              value={awsRegion}
              onChange={(e) => setAwsRegion(e.target.value)}
              className={styles.select}
            >
              <option value="us-east-1">US East (N. Virginia)</option>
              <option value="us-west-2">US West (Oregon)</option>
              <option value="eu-west-1">EU (Ireland)</option>
              <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
            </select>
          </div>
          <p className={styles.warning}>
            ⚠️ Your AWS credentials will be encrypted and stored securely. They will only be used for deployment.
          </p>
        </div>
      )}

      <div className={styles.actions}>
        <button onClick={handleDeploy} className={styles.deployButton}>
          Deploy Model
        </button>
      </div>
    </div>
  );
}

