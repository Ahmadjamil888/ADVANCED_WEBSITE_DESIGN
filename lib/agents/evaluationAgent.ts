// Evaluation Agent - Handles model evaluation, metrics calculation, and performance analysis
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { supabase } from '../db/client';

export class EvaluationAgent {
  private llm: ChatOpenAI;

  constructor() {
    this.llm = new ChatOpenAI({
      modelName: 'gpt-4',
      temperature: 0.1,
    });
  }

  // Evaluate trained model
  async evaluateModel(trainingRunId: string) {
    try {
      // Get training run and related data
      const { data: trainingRun } = await supabase
        .from('training_runs')
        .select('*, models(*), datasets(*)')
        .eq('id', trainingRunId)
        .single();

      // Get training metrics
      const { data: metrics } = await supabase
        .from('metrics')
        .select('*')
        .eq('training_run_id', trainingRunId)
        .order('epoch');

      // Perform comprehensive evaluation
      const evaluation = await this.performEvaluation(trainingRun, metrics);

      // Store evaluation results
      await supabase
        .from('training_runs')
        .update({
          metrics: {
            ...trainingRun.metrics,
            evaluation,
          },
        })
        .eq('id', trainingRunId);

      // Generate evaluation report
      const report = await this.generateEvaluationReport(trainingRun, evaluation);

      return {
        evaluation,
        report,
      };
    } catch (error) {
      console.error('Model Evaluation Error:', error);
      throw error;
    }
  }

  // Perform comprehensive evaluation
  private async performEvaluation(trainingRun: any, metrics: any[]) {
    const evaluationPrompt = `Perform comprehensive evaluation of this trained model:
    Training Run: ${JSON.stringify(trainingRun)}
    Metrics History: ${JSON.stringify(metrics)}

    Evaluate:
    - Performance metrics (accuracy, precision, recall, F1, AUC)
    - Learning curve analysis
    - Overfitting/underfitting detection
    - Convergence quality
    - Model robustness
    - Computational efficiency

    Return detailed JSON evaluation with scores and analysis.`;

    const response = await this.llm.invoke([
      new SystemMessage('You are an expert model evaluator and ML metrics specialist.'),
      new HumanMessage(evaluationPrompt),
    ]);

    const evaluation = JSON.parse(response.content as string);

    // Calculate additional metrics
    evaluation.calculatedMetrics = this.calculateMetrics(metrics);
    evaluation.healthScore = this.calculateHealthScore(evaluation, metrics);
    evaluation.learningCurveAnalysis = this.analyzeLearningCurve(metrics);

    return evaluation;
  }

  // Calculate standard ML metrics
  private calculateMetrics(metrics: any[]) {
    const lossMetrics = metrics.filter(m => m.metric_name === 'loss');
    const accuracyMetrics = metrics.filter(m => m.metric_name === 'accuracy');

    const finalLoss = lossMetrics[lossMetrics.length - 1]?.metric_value || 0;
    const finalAccuracy = accuracyMetrics[accuracyMetrics.length - 1]?.metric_value || 0;
    const bestAccuracy = Math.max(...accuracyMetrics.map(m => m.metric_value));

    // Calculate learning curve smoothness
    const lossValues = lossMetrics.map(m => m.metric_value);
    const accuracyValues = accuracyMetrics.map(m => m.metric_value);

    const lossVariance = this.calculateVariance(lossValues);
    const accuracyVariance = this.calculateVariance(accuracyValues);

    return {
      finalLoss,
      finalAccuracy,
      bestAccuracy,
      lossVariance,
      accuracyVariance,
      totalEpochs: Math.max(...metrics.map(m => m.epoch || 0)),
    };
  }

  // Calculate health score (0-100)
  private calculateHealthScore(evaluation: any, metrics: any[]) {
    let score = 100;

    const finalAccuracy = evaluation.calculatedMetrics?.finalAccuracy || 0;
    const finalLoss = evaluation.calculatedMetrics?.finalLoss || 0;
    const lossVariance = evaluation.calculatedMetrics?.lossVariance || 0;
    const accuracyVariance = evaluation.calculatedMetrics?.accuracyVariance || 0;

    // Penalize low accuracy
    if (finalAccuracy < 0.6) score -= (0.6 - finalAccuracy) * 100;
    if (finalAccuracy < 0.8) score -= 20;

    // Penalize high final loss
    if (finalLoss > 0.5) score -= (finalLoss - 0.5) * 50;

    // Penalize high variance (unstable training)
    if (lossVariance > 0.1) score -= lossVariance * 200;
    if (accuracyVariance > 0.05) score -= accuracyVariance * 500;

    // Bonus for low variance (stable training)
    if (accuracyVariance < 0.05) score += 5;
    if (lossVariance < 0.1) score += 5;

    return Math.max(0, Math.min(100, score));
  }

  // Analyze learning curve
  private analyzeLearningCurve(metrics: any[]) {
    const lossMetrics = metrics.filter(m => m.metric_name === 'loss');
    const accuracyMetrics = metrics.filter(m => m.metric_name === 'accuracy');

    const analysis = {
      convergence: 'unknown',
      overfitting: false,
      underfitting: false,
      stability: 'stable',
    };

    if (lossMetrics.length >= 3) {
      const recentLoss = lossMetrics.slice(-3).map(m => m.metric_value);
      const earlyLoss = lossMetrics.slice(0, 3).map(m => m.metric_value);

      // Check convergence (loss decreasing and stabilizing)
      const isDecreasing = recentLoss.every((loss, i) =>
        i === 0 || loss <= recentLoss[i-1] * 1.05 // Allow slight increases
      );

      if (isDecreasing && recentLoss[recentLoss.length - 1] < earlyLoss[0] * 0.5) {
        analysis.convergence = 'good';
      } else if (isDecreasing) {
        analysis.convergence = 'slow';
      } else {
        analysis.convergence = 'poor';
      }

      // Check for overfitting (training loss much lower than expected)
      const avgRecentLoss = recentLoss.reduce((a, b) => a + b, 0) / recentLoss.length;
      if (avgRecentLoss < 0.1 && analysis.convergence === 'good') {
        analysis.overfitting = true;
      }

      // Check for underfitting (high loss throughout)
      if (avgRecentLoss > 1.0) {
        analysis.underfitting = true;
      }
    }

    // Check stability
    const accuracyValues = accuracyMetrics.map(m => m.metric_value);
    const accuracyVariance = this.calculateVariance(accuracyValues);

    if (accuracyVariance > 0.1) {
      analysis.stability = 'unstable';
    } else if (accuracyVariance > 0.05) {
      analysis.stability = 'moderate';
    }

    return analysis;
  }

  // Calculate variance
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;

    return variance;
  }

  // Generate evaluation report
  private async generateEvaluationReport(trainingRun: any, evaluation: any) {
    const reportPrompt = `Generate a comprehensive evaluation report for this model:
    Training Run: ${JSON.stringify(trainingRun)}
    Evaluation Results: ${JSON.stringify(evaluation)}

    Include:
    - Executive summary
    - Performance metrics analysis
    - Learning curve insights
    - Strengths and weaknesses
    - Recommendations for improvement
    - Deployment readiness assessment

    Format as markdown report.`;

    const response = await this.llm.invoke([
      new SystemMessage('You are a technical report writer specializing in ML model evaluation.'),
      new HumanMessage(reportPrompt),
    ]);

    return response.content as string;
  }

  // Compare models
  async compareModels(trainingRunIds: string[]) {
    try {
      const evaluations = [];

      for (const runId of trainingRunIds) {
        const evaluation = await this.evaluateModel(runId);
        evaluations.push({
          trainingRunId: runId,
          evaluation: evaluation.evaluation,
        });
      }

      // Generate comparison report
      const comparison = await this.generateComparisonReport(evaluations);

      return {
        evaluations,
        comparison,
      };
    } catch (error) {
      console.error('Model Comparison Error:', error);
      throw error;
    }
  }

  // Generate comparison report
  private async generateComparisonReport(evaluations: any[]) {
    const comparisonPrompt = `Compare these model evaluations:
    Evaluations: ${JSON.stringify(evaluations)}

    Provide:
    - Performance comparison
    - Best model identification
    - Trade-off analysis
    - Recommendations for model selection

    Return structured comparison.`;

    const response = await this.llm.invoke([
      new SystemMessage('You are an expert in comparative model evaluation.'),
      new HumanMessage(comparisonPrompt),
    ]);

    return JSON.parse(response.content as string);
  }

  // Get evaluation history
  async getEvaluationHistory(modelId: string) {
    try {
      const { data: trainingRuns } = await supabase
        .from('training_runs')
        .select('id, created_at, metrics')
        .eq('model_id', modelId)
        .order('created_at', { ascending: false });

      const history = trainingRuns.map(run => ({
        trainingRunId: run.id,
        date: run.created_at,
        evaluation: run.metrics?.evaluation,
      }));

      return history;
    } catch (error) {
      console.error('Get Evaluation History Error:', error);
      throw error;
    }
  }

  // Validate evaluation metrics
  async validateMetrics(trainingRunId: string) {
    try {
      const { data: metrics } = await supabase
        .from('metrics')
        .select('*')
        .eq('training_run_id', trainingRunId);

      const validation = {
        isValid: true,
        issues: [],
        recommendations: [],
      };

      // Check for missing metrics
      const hasLoss = metrics.some(m => m.metric_name === 'loss');
      const hasAccuracy = metrics.some(m => m.metric_name === 'accuracy');

      if (!hasLoss) {
        validation.issues.push('Missing loss metrics');
        validation.isValid = false;
      }

      if (!hasAccuracy) {
        validation.issues.push('Missing accuracy metrics');
        validation.isValid = false;
      }

      // Check for invalid values
      const invalidMetrics = metrics.filter(m =>
        !isFinite(m.metric_value) || m.metric_value < 0 || m.metric_value > 1
      );

      if (invalidMetrics.length > 0) {
        validation.issues.push(`${invalidMetrics.length} metrics have invalid values`);
        validation.isValid = false;
        validation.recommendations.push('Check metric calculation logic');
      }

      return validation;
    } catch (error) {
      console.error('Metrics Validation Error:', error);
      throw error;
    }
  }
}

export const evaluationAgent = new EvaluationAgent();
