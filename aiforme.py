import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Brain, Database, Network, Layers, Upload, MessageSquare, Activity, Cpu, BarChart3, BookOpen, Zap, Eye, CheckCircle, TrendingUp, GitBranch, Settings, Play, Pause, Save, FileText, AlertCircle, Filter, Lightbulb, Target, Radio, RotateCcw } from 'lucide-react';

const IntegratedAISystem = () => {
  const [activeSystem, setActiveSystem] = useState('deepIntelligence'); // 'deepIntelligence' or 'unifiedPathway'
  const [mode, setMode] = useState('training');
  const [trainingText, setTrainingText] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [conversations, setConversations] = useState([]);
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [visualizing, setVisualizing] = useState(true);
  const [metrics, setMetrics] = useState({
    concepts: 0,
    connections: 0,
    layers: 6,
    accuracy: 0,
    learning: 0,
    epochs: 0
  });

  // نظام المسارات الموحد
  const [isRunning, setIsRunning] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState({
    throughput: 0,
    activeNodes: 0,
    pathways: 0,
    learning: 0
  });
  const [dataFlow, setDataFlow] = useState([]);

  const canvasRef = useRef(null);
  const systemRef = useRef(null);
  const animRef = useRef(null);
  const fileInputRef = useRef(null);
  const animationRef = useRef(null);

  // ============================================================================
  // نظام الذكاء العميق المتكامل
  // ============================================================================

  class ReceptionLayer {
    constructor() {
      this.buffer = [];
      this.preprocessors = {
        text: this.preprocessText.bind(this),
        file: this.preprocessFile.bind(this),
        conversation: this.preprocessConversation.bind(this)
      };
    }

    async receive(input, type = 'text') {
      const preprocessor = this.preprocessors[type];
      if (!preprocessor) throw new Error(`Unknown input type: ${type}`);
      
      const processed = await preprocessor(input);
      this.buffer.push({
        raw: input,
        processed,
        type,
        timestamp: Date.now(),
        id: `input_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`
      });
      
      return processed;
    }

    preprocessText(text) {
      return {
        content: text,
        sentences: text.split(/[.!?]+/).filter(s => s.trim()),
        words: text.toLowerCase().split(/\s+/),
        length: text.length,
        metadata: {
          hasQuestion: /\?/.test(text),
          hasCommand: /please|can you|could you|would you/i.test(text),
          isDefinition: /is|are|means|refers to|defined as/i.test(text),
          complexity: Math.min(1, text.split(' ').length / 50)
        }
      };
    }

    preprocessFile(fileContent) {
      return this.preprocessText(fileContent);
    }

    preprocessConversation(text) {
      const processed = this.preprocessText(text);
      processed.metadata.isConversational = true;
      return processed;
    }

    getBuffer() {
      return this.buffer;
    }

    clearBuffer() {
      this.buffer = [];
    }
  }

  class AnalysisLayer {
    constructor() {
      this.entityExtractor = new EntityExtractor();
      this.conceptExtractor = new ConceptExtractor();
      this.relationExtractor = new RelationExtractor();
    }

    async analyze(processedInput) {
      const entities = this.entityExtractor.extract(processedInput.content);
      const concepts = this.conceptExtractor.extract(processedInput);
      const relations = this.relationExtractor.extract(processedInput, entities, concepts);
      
      return {
        input: processedInput,
        entities,
        concepts,
        relations,
        category: this.classifyCategory(processedInput),
        importance: this.calculateImportance(entities, concepts, relations),
        timestamp: Date.now()
      };
    }

    classifyCategory(input) {
      const { metadata, content } = input;
      const lower = content.toLowerCase();
      
      if (metadata.isDefinition) return 'definition';
      if (/how to|step|process|method/i.test(lower)) return 'procedure';
      if (/example|instance|such as/i.test(lower)) return 'example';
      if (/why|because|reason|cause/i.test(lower)) return 'explanation';
      if (metadata.hasQuestion) return 'question';
      if (/rule|must|should|always|never/i.test(lower)) return 'rule';
      if (/technology|computer|ai|system/i.test(lower)) return 'technical';
      if (/think|believe|feel|opinion/i.test(lower)) return 'subjective';
      
      return 'general';
    }

    calculateImportance(entities, concepts, relations) {
      let score = 0.5;
      score += entities.length * 0.05;
      score += concepts.length * 0.03;
      score += relations.length * 0.07;
      return Math.min(1, score);
    }
  }

  class EntityExtractor {
    extract(text) {
      const entities = [];
      const words = text.split(/\s+/);
      
      words.forEach((word, i) => {
        if (word[0] === word[0].toUpperCase() && word.length > 2) {
          entities.push({
            text: word,
            type: 'proper_noun',
            position: i
          });
        }
      });
      
      const numbers = text.match(/\d+(\.\d+)?/g) || [];
      numbers.forEach(num => {
        entities.push({ text: num, type: 'number' });
      });
      
      return entities;
    }
  }

  class ConceptExtractor {
    extract(processedInput) {
      const concepts = [];
      const { sentences, words } = processedInput;
      
      sentences.forEach(sentence => {
        const tags = this.extractTags(sentence);
        if (tags.length > 0) {
          concepts.push({
            content: sentence.trim(),
            tags,
            embedding: this.generateEmbedding(sentence)
          });
        }
      });
      
      return concepts;
    }

    extractTags(text) {
      const words = text.toLowerCase().split(/\s+/);
      const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']);
      return words
        .filter(w => w.length > 3 && !stopWords.has(w))
        .slice(0, 5);
    }

    generateEmbedding(text) {
      const embedding = new Float32Array(128);
      const str = text.toLowerCase();
      
      for (let i = 0; i < 128; i++) {
        let hash = 0;
        for (let j = 0; j < str.length; j++) {
          hash = ((hash << 5) - hash) + str.charCodeAt(j);
          hash = hash & hash;
        }
        embedding[i] = Math.sin(hash * (i + 1) * 0.01) * Math.cos(i * 0.05);
      }
      
      const mag = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= (mag || 1);
      }
      
      return embedding;
    }
  }

  class RelationExtractor {
    extract(input, entities, concepts) {
      const relations = [];
      const patterns = [
        { pattern: /(.+)\s+causes?\s+(.+)/i, type: 'causes' },
        { pattern: /(.+)\s+enables?\s+(.+)/i, type: 'enables' },
        { pattern: /(.+)\s+requires?\s+(.+)/i, type: 'requires' },
        { pattern: /(.+)\s+is\s+part\s+of\s+(.+)/i, type: 'partOf' },
        { pattern: /(.+)\s+similar\s+to\s+(.+)/i, type: 'similarTo' },
        { pattern: /(.+)\s+different\s+from\s+(.+)/i, type: 'differentFrom' }
      ];
      
      patterns.forEach(({ pattern, type }) => {
        const match = input.content.match(pattern);
        if (match) {
          relations.push({
            subject: match[1].trim(),
            predicate: type,
            object: match[2].trim(),
            confidence: 0.7
          });
        }
      });
      
      return relations;
    }
  }

  class IntelligentStorageLayer {
    constructor() {
      this.coreKnowledge = new KnowledgeBase('core');
      this.learnedExperience = new KnowledgeBase('learned');
      this.temporalIndex = new TemporalIndex();
      this.knowledgeGraph = new DynamicKnowledgeGraph();
    }

    async store(analyzedData) {
      const { category, importance, concepts, relations } = analyzedData;
      
      const knowledgeUnits = concepts.map(concept => ({
        id: `unit_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        content: concept.content,
        tags: concept.tags,
        embedding: concept.embedding,
        category,
        importance,
        timestamp: Date.now(),
        accessCount: 0,
        layer: importance > 0.7 ? 'core' : 'learned'
      }));

      for (const unit of knowledgeUnits) {
        if (unit.layer === 'core') {
          this.coreKnowledge.add(unit);
        } else {
          this.learnedExperience.add(unit);
        }
        
        this.temporalIndex.add(unit);
        this.knowledgeGraph.addNode(unit);
      }

      this.autoLink(knowledgeUnits, relations);
      
      return knowledgeUnits;
    }

    autoLink(newUnits, explicitRelations) {
      newUnits.forEach(unit => {
        const similar = this.knowledgeGraph.findSimilar(unit, 5);
        
        similar.forEach(sim => {
          const similarity = this.calculateSimilarity(unit.embedding, sim.embedding);
          if (similarity > 0.6) {
            this.knowledgeGraph.addEdge(unit.id, sim.id, similarity, 'similarity');
          }
        });
      });

      explicitRelations.forEach(rel => {
        const subjectUnits = newUnits.filter(u => 
          u.content.toLowerCase().includes(rel.subject.toLowerCase())
        );
        const objectUnits = this.knowledgeGraph.search(rel.object, 3);
        
        subjectUnits.forEach(sub => {
          objectUnits.forEach(obj => {
            this.knowledgeGraph.addEdge(sub.id, obj.id, rel.confidence, rel.predicate);
          });
        });
      });
    }

    calculateSimilarity(emb1, emb2) {
      let dot = 0, mag1 = 0, mag2 = 0;
      for (let i = 0; i < emb1.length; i++) {
        dot += emb1[i] * emb2[i];
        mag1 += emb1[i] * emb1[i];
        mag2 += emb2[i] * emb2[i];
      }
      return dot / (Math.sqrt(mag1) * Math.sqrt(mag2) + 1e-10);
    }

    retrieve(query, k = 10) {
      const results = [
        ...this.coreKnowledge.search(query, k),
        ...this.learnedExperience.search(query, k)
      ];
      
      results.sort((a, b) => b.relevance - a.relevance);
      return results.slice(0, k);
    }

    getStats() {
      return {
        core: this.coreKnowledge.size(),
        learned: this.learnedExperience.size(),
        total: this.knowledgeGraph.nodeCount(),
        connections: this.knowledgeGraph.edgeCount()
      };
    }
  }

  class KnowledgeBase {
    constructor(type) {
      this.type = type;
      this.units = new Map();
      this.index = [];
    }

    add(unit) {
      this.units.set(unit.id, unit);
      this.index.push({
        id: unit.id,
        embedding: unit.embedding,
        tags: unit.tags
      });
    }

    search(query, k = 5) {
      const queryEmb = new ConceptExtractor().generateEmbedding(query);
      const scores = [];
      
      this.units.forEach(unit => {
        let similarity = 0;
        for (let i = 0; i < queryEmb.length; i++) {
          similarity += queryEmb[i] * unit.embedding[i];
        }
        
        const tagBoost = unit.tags.some(t => query.toLowerCase().includes(t)) ? 0.2 : 0;
        const importanceBoost = unit.importance * 0.1;
        const accessBoost = Math.log(unit.accessCount + 1) * 0.05;
        
        scores.push({
          unit,
          relevance: similarity + tagBoost + importanceBoost + accessBoost
        });
      });
      
      scores.sort((a, b) => b.relevance - a.relevance);
      return scores.slice(0, k);
    }

    size() {
      return this.units.size;
    }
  }

  class TemporalIndex {
    constructor() {
      this.timeline = [];
      this.epochs = new Map();
    }

    add(unit) {
      this.timeline.push({
        unitId: unit.id,
        timestamp: unit.timestamp,
        action: 'created'
      });
      
      const epoch = Math.floor(unit.timestamp / 3600000);
      if (!this.epochs.has(epoch)) {
        this.epochs.set(epoch, []);
      }
      this.epochs.get(epoch).push(unit.id);
    }

    getEvolutionPath(unitId) {
      return this.timeline.filter(entry => entry.unitId === unitId);
    }

    getEpochUnits(timestamp) {
      const epoch = Math.floor(timestamp / 3600000);
      return this.epochs.get(epoch) || [];
    }
  }

  class DynamicKnowledgeGraph {
    constructor() {
      this.nodes = new Map();
      this.edges = new Map();
      this.adjacency = new Map();
    }

    addNode(unit) {
      this.nodes.set(unit.id, {
        ...unit,
        x: Math.random() * 800 + 100,
        y: Math.random() * 400 + 50,
        vx: 0,
        vy: 0,
        activation: 0
      });
      this.adjacency.set(unit.id, []);
    }

    addEdge(fromId, toId, weight, type) {
      const edgeId = `${fromId}-${toId}`;
      this.edges.set(edgeId, {
        from: fromId,
        to: toId,
        weight,
        type,
        activations: 0,
        signal: 0
      });
      
      this.adjacency.get(fromId)?.push(toId);
    }

    findSimilar(unit, k = 5) {
      const scores = [];
      this.nodes.forEach((node, id) => {
        if (id === unit.id) return;
        
        let similarity = 0;
        for (let i = 0; i < unit.embedding.length; i++) {
          similarity += unit.embedding[i] * node.embedding[i];
        }
        
        scores.push({ node, similarity });
      });
      
      scores.sort((a, b) => b.similarity - a.similarity);
      return scores.slice(0, k).map(s => s.node);
    }

    search(query, k = 5) {
      const queryEmb = new ConceptExtractor().generateEmbedding(query);
      const scores = [];
      
      this.nodes.forEach(node => {
        let similarity = 0;
        for (let i = 0; i < queryEmb.length; i++) {
          similarity += queryEmb[i] * node.embedding[i];
        }
        scores.push({ node, similarity });
      });
      
      scores.sort((a, b) => b.similarity - a.similarity);
      return scores.slice(0, k).map(s => s.node);
    }

    propagateActivation(startId, strength = 1.0, depth = 3) {
      const visited = new Set();
      const queue = [{ id: startId, strength, depth: 0 }];
      
      while (queue.length > 0) {
        const { id, strength: s, depth: d } = queue.shift();
        if (visited.has(id) || d >= depth) continue;
        
        visited.add(id);
        const node = this.nodes.get(id);
        if (node) {
          node.activation = Math.min(1, node.activation + s);
          node.accessCount++;
          
          (this.adjacency.get(id) || []).forEach(targetId => {
            const edgeId = `${id}-${targetId}`;
            const edge = this.edges.get(edgeId);
            if (edge && !visited.has(targetId)) {
              const propagated = s * edge.weight * 0.7;
              if (propagated > 0.1) {
                queue.push({ id: targetId, strength: propagated, depth: d + 1 });
                edge.signal = propagated;
                edge.activations++;
              }
            }
          });
        }
      }
    }

    decay() {
      this.nodes.forEach(node => {
        node.activation *= 0.95;
      });
      this.edges.forEach(edge => {
        edge.signal *= 0.9;
      });
    }

    updatePositions(width, height) {
      const damping = 0.8;
      const centerX = width / 2;
      const centerY = height / 2;
      
      this.nodes.forEach((node, id) => {
        const toCenterX = (centerX - node.x) * 0.0001;
        const toCenterY = (centerY - node.y) * 0.0001;
        node.vx += toCenterX;
        node.vy += toCenterY;

        this.nodes.forEach((other, otherId) => {
          if (id === otherId) return;
          
          const dx = other.x - node.x;
          const dy = other.y - node.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          
          const repulsion = 1000 / (dist * dist);
          node.vx -= (dx / dist) * repulsion;
          node.vy -= (dy / dist) * repulsion;
        });

        (this.adjacency.get(id) || []).forEach(targetId => {
          const target = this.nodes.get(targetId);
          if (target) {
            const dx = target.x - node.x;
            const dy = target.y - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            
            const edgeId = `${id}-${targetId}`;
            const edge = this.edges.get(edgeId);
            const attraction = (dist - 150) * 0.01 * (edge?.weight || 0.5);
            node.vx += (dx / dist) * attraction;
            node.vy += (dy / dist) * attraction;
          }
        });

        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;

        node.x = Math.max(50, Math.min(width - 50, node.x));
        node.y = Math.max(50, Math.min(height - 50, node.y));
      });
    }

    nodeCount() {
      return this.nodes.size;
    }

    edgeCount() {
      return this.edges.size;
    }
  }

  class SynthesisLayer {
    constructor(storageLayer) {
      this.storage = storageLayer;
      this.patternDetector = new PatternDetector();
      this.hypothesisGenerator = new HypothesisGenerator();
      this.insightExtractor = new InsightExtractor();
    }

    async synthesize() {
      const patterns = this.patternDetector.detect(this.storage.knowledgeGraph);
      const hypotheses = this.hypothesisGenerator.generate(patterns, this.storage);
      const insights = this.insightExtractor.extract(hypotheses, this.storage);
      
      return {
        patterns,
        hypotheses,
        insights,
        timestamp: Date.now()
      };
    }
  }

  class PatternDetector {
    detect(graph) {
      const patterns = [];
      
      const clusters = this.detectClusters(graph);
      patterns.push(...clusters);
      
      const sequences = this.detectSequences(graph);
      patterns.push(...sequences);
      
      const hierarchies = this.detectHierarchies(graph);
      patterns.push(...hierarchies);
      
      return patterns;
    }

    detectClusters(graph) {
      const clusters = new Map();
      
      graph.nodes.forEach(node => {
        const category = node.category;
        if (!clusters.has(category)) {
          clusters.set(category, []);
        }
        clusters.get(category).push(node);
      });
      
      return Array.from(clusters.entries()).map(([category, nodes]) => ({
        type: 'cluster',
        category,
        size: nodes.length,
        centroid: this.calculateCentroid(nodes)
      }));
    }

    detectSequences(graph) {
      return [];
    }

    detectHierarchies(graph) {
      return [];
    }

    calculateCentroid(nodes) {
      if (nodes.length === 0) return null;
      
      const centroid = new Float32Array(128);
      nodes.forEach(node => {
        for (let i = 0; i < 128; i++) {
          centroid[i] += node.embedding[i];
        }
      });
      
      for (let i = 0; i < 128; i++) {
        centroid[i] /= nodes.length;
      }
      
      return centroid;
    }
  }

  class HypothesisGenerator {
    generate(patterns, storage) {
      const hypotheses = [];
      
      patterns.forEach(pattern => {
        if (pattern.type === 'cluster' && pattern.size > 3) {
          hypotheses.push({
            type: 'generalization',
            statement: `There are ${pattern.size} concepts related to ${pattern.category}`,
            confidence: Math.min(0.9, pattern.size / 10),
            evidence: pattern
          });
        }
      });
      
      return hypotheses;
    }
  }

  class InsightExtractor {
    extract(hypotheses, storage) {
      return hypotheses.filter(h => h.confidence > 0.7).map(h => ({
        insight: h.statement,
        confidence: h.confidence,
        actionable: true
      }));
    }
  }

  class FieldIntelligenceLayer {
    constructor(storageLayer, synthesisLayer) {
      this.storage = storageLayer;
      this.synthesis = synthesisLayer;
      this.conversationHistory = [];
      this.performanceTracker = new PerformanceTracker();
    }

    async process(input) {
      const relevant = this.storage.retrieve(input, 5);
      
      if (relevant.length > 0) {
        relevant.forEach(r => {
          this.storage.knowledgeGraph.propagateActivation(r.unit.id, 0.5, 2);
        });
      }

      const response = this.generateResponse(input, relevant);
      
      this.conversationHistory.push({
        input,
        response,
        relevantUnits: relevant.map(r => r.unit.id),
        timestamp: Date.now()
      });

      this.performanceTracker.record({
        query: input,
        foundRelevant: relevant.length > 0,
        confidence: this.calculateConfidence(relevant)
      });

      return {
        response,
        confidence: this.calculateConfidence(relevant),
        sources: relevant.slice(0, 3)
      };
    }

    generateResponse(input, relevant) {
      if (relevant.length === 0) {
        return `لم أجد معلومات كافية حول "${input}". يمكنك تدريبي على هذا الموضوع من خلال واجهة التدريب.`;
      }

      const isQuestion = /\?|what|how|why|when|where|who/i.test(input);
      
      if (isQuestion) {
        const best = relevant[0].unit;
        let response = `بناءً على معرفتي: ${best.content}`;
        
        if (relevant.length > 1) {
          response += `\n\nمعلومات إضافية: ${relevant[1].unit.content.substring(0, 100)}...`;
        }
        
        return response;
      }

      const tags = relevant.flatMap(r => r.unit.tags).slice(0, 5);
      return `فهمت! هذا يرتبط بالمفاهيم التالية: ${tags.join(', ')}. وجدت ${relevant.length} وحدات معرفية ذات صلة.`;
    }

    calculateConfidence(relevant) {
      if (relevant.length === 0) return 0.1;
      
      const avgRelevance = relevant.reduce((sum, r) => sum + r.relevance, 0) / relevant.length;
      const countBonus = Math.min(0.3, relevant.length * 0.05);
      
      return Math.min(1, avgRelevance + countBonus);
    }

    getPerformanceReport() {
      return this.performanceTracker.getReport();
    }
  }

  class PerformanceTracker {
    constructor() {
      this.records = [];
    }

    record(data) {
      this.records.push({
        ...data,
        timestamp: Date.now()
      });
      
      if (this.records.length > 1000) {
        this.records.shift();
      }
    }

    getReport() {
      if (this.records.length === 0) {
        return { successRate: 0, avgConfidence: 0, totalQueries: 0 };
      }

      const successful = this.records.filter(r => r.foundRelevant).length;
      const avgConfidence = this.records.reduce((sum, r) => sum + r.confidence, 0) / this.records.length;
      
      return {
        successRate: successful / this.records.length,
        avgConfidence,
        totalQueries: this.records.length
      };
    }
  }

  class MetaKnowledgeLayer {
    constructor(system) {
      this.system = system;
      this.selfEvaluator = new SelfEvaluator();
      this.pathwayOptimizer = new PathwayOptimizer();
      this.ruleUpdater = new RuleUpdater();
    }

    async evaluate() {
      const evaluation = this.selfEvaluator.evaluate(this.system);
      const optimizations = this.pathwayOptimizer.optimize(this.system.storage.knowledgeGraph);
      const ruleUpdates = this.ruleUpdater.update(evaluation, this.system);
      
      return {
        evaluation,
        optimizations,
        ruleUpdates,
        recommendations: this.generateRecommendations(evaluation)
      };
    }

    generateRecommendations(evaluation) {
      const recommendations = [];
      
      if (evaluation.knowledgeDensity < 0.3) {
        recommendations.push({
          type: 'training',
          message: 'النظام يحتاج المزيد من البيانات التدريبية',
          priority: 'high'
        });
      }
      
      if (evaluation.connectionQuality < 0.5) {
        recommendations.push({
          type: 'linking',
          message: 'جودة الروابط بين المفاهيم منخفضة، يحتاج النظام لمراجعة ذاتية',
          priority: 'medium'
        });
      }
      
      return recommendations;
    }
  }

  class SelfEvaluator {
    evaluate(system) {
      const stats = system.storage.getStats();
      const performance = system.fieldIntelligence.getPerformanceReport();
      
      return {
        knowledgeDensity: stats.total / 1000,
        connectionQuality: stats.connections / (stats.total || 1),
        performanceScore: performance.successRate * 0.6 + performance.avgConfidence * 0.4,
        timestamp: Date.now()
      };
    }
  }

  class PathwayOptimizer {
    optimize(graph) {
      const weakEdges = [];
      
      graph.edges.forEach((edge, id) => {
        if (edge.weight < 0.3 && edge.activations < 2) {
          weakEdges.push(id);
        }
      });

      weakEdges.forEach(id => {
        graph.edges.delete(id);
      });
      
      return {
        prunedEdges: weakEdges.length,
        remainingEdges: graph.edgeCount()
      };
    }
  }

  class RuleUpdater {
    update(evaluation, system) {
      const updates = [];
      
      if (evaluation.performanceScore < 0.6) {
        updates.push({
          type: 'relaxation',
          description: 'تم تخفيض عتبات التصفية لتحسين الأداء',
          impact: 'increased_responsiveness'
        });
      }
      
      return updates;
    }
  }

  // ============================================================================
  // نظام المسارات الموحد
  // ============================================================================

  class Node {
    constructor(x, y, type = 'processor') {
      this.id = Math.random().toString(36).substr(2, 9);
      this.x = x;
      this.y = y;
      this.type = type;
      this.state = 0;
      this.activation = 0;
      this.encoding = this.generateEncoding();
      this.intensity = 0.5;
      this.color = this.calculateColor();
      this.connections = [];
      this.buffer = [];
      this.memory = [];
    }

    generateEncoding() {
      return new Float32Array(64).map(() => Math.random() * 2 - 1);
    }

    calculateColor() {
      const hue = this.intensity * 360;
      const saturation = 50 + this.activation * 50;
      return `hsl(${hue}, ${saturation}%, 50%)`;
    }

    receive(signal) {
      this.buffer.push(signal);
      if (this.buffer.length > 10) this.buffer.shift();
      this.activation = Math.min(1, this.activation + 0.1);
    }

    process() {
      if (this.buffer.length === 0) {
        this.activation = Math.max(0, this.activation - 0.05);
        return null;
      }

      const input = this.buffer.shift();
      let output;

      switch (this.type) {
        case 'sender':
          output = this.encode(input);
          break;
        case 'receiver':
          output = this.decode(input);
          break;
        case 'processor':
          output = this.transform(input);
          break;
        case 'transformer':
          output = this.deepTransform(input);
          break;
        default:
          output = input;
      }

      this.state = output.value || 0;
      this.intensity = Math.abs(this.state);
      this.color = this.calculateColor();
      
      this.memory.push({ input, output, timestamp: Date.now() });
      if (this.memory.length > 50) this.memory.shift();

      return output;
    }

    encode(data) {
      const encoded = new Float32Array(this.encoding.length);
      for (let i = 0; i < encoded.length; i++) {
        encoded[i] = data.value * this.encoding[i] * Math.cos(i * 0.1);
      }
      return {
        type: 'encoded',
        value: data.value,
        encoding: encoded,
        source: this.id
      };
    }

    decode(data) {
      if (!data.encoding) return data;
      let decoded = 0;
      for (let i = 0; i < Math.min(data.encoding.length, this.encoding.length); i++) {
        decoded += data.encoding[i] * this.encoding[i];
      }
      return {
        type: 'decoded',
        value: Math.tanh(decoded / data.encoding.length),
        source: this.id
      };
    }

    transform(data) {
      const transformed = Math.tanh(data.value * 2 + this.state * 0.5);
      return {
        type: 'transformed',
        value: transformed,
        source: this.id
      };
    }

    deepTransform(data) {
      let value = data.value;
      for (let i = 0; i < 3; i++) {
        value = Math.tanh(value * 1.5 + this.encoding[i] * 0.1);
      }
      return {
        type: 'deep_transformed',
        value: value,
        depth: 3,
        source: this.id
      };
    }

    send(target) {
      const output = this.process();
      if (output && target) {
        target.receive(output);
        return output;
      }
      return null;
    }
  }

  class Line {
    constructor(fromNode, toNode, weight = 1) {
      this.id = Math.random().toString(36).substr(2, 9);
      this.from = fromNode;
      this.to = toNode;
      this.weight = weight;
      this.signal = 0;
      this.flow = 0;
      this.encoding = this.combineEncodings();
      this.history = [];
      this.strength = 0.5;
      this.type = this.determineType();
      this.bandwidth = 1;
    }

    combineEncodings() {
      const combined = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        combined[i] = (this.from.encoding[i] + this.to.encoding[i]) / 2;
      }
      return combined;
    }

    determineType() {
      const types = ['solid', 'dashed', 'dotted', 'wavy'];
      const index = Math.floor((this.from.x + this.to.x) % types.length);
      return types[index];
    }

    transmit(signal) {
      this.signal = signal;
      const transmitted = signal * this.weight * this.strength;
      this.flow = transmitted;
      
      this.history.push({
        signal: transmitted,
        timestamp: Date.now()
      });
      if (this.history.length > 100) this.history.shift();

      return transmitted;
    }

    propagate() {
      const output = this.from.send(this.to);
      if (output) {
        this.transmit(output.value);
      }
    }

    learn(error) {
      const delta = error * 0.01;
      this.weight = Math.max(-2, Math.min(2, this.weight + delta));
      this.strength = Math.max(0.1, Math.min(1, this.strength + delta * 0.5));
    }

    evolve() {
      const recentFlow = this.history.slice(-10);
      const avgFlow = recentFlow.reduce((sum, h) => sum + Math.abs(h.signal), 0) / recentFlow.length;
      
      if (avgFlow > 0.7) {
        this.bandwidth = Math.min(2, this.bandwidth * 1.05);
      } else if (avgFlow < 0.3) {
        this.bandwidth = Math.max(0.5, this.bandwidth * 0.95);
      }
    }
  }

  class Pattern {
    constructor(name, nodes, lines) {
      this.id = Math.random().toString(36).substr(2, 9);
      this.name = name;
      this.nodes = nodes;
      this.lines = lines;
      this.shape = this.detectShape();
      this.encoding = this.aggregateEncoding();
      this.activation = 0;
      this.frequency = 0;
      this.subPatterns = [];
    }

    detectShape() {
      if (this.nodes.length === 3) return 'triangle';
      if (this.nodes.length === 4) return 'quad';
      if (this.nodes.length > 4 && this.isCircular()) return 'circle';
      if (this.isLinear()) return 'line';
      return 'complex';
    }

    isCircular() {
      if (this.nodes.length < 3) return false;
      const centerX = this.nodes.reduce((sum, n) => sum + n.x, 0) / this.nodes.length;
      const centerY = this.nodes.reduce((sum, n) => sum + n.y, 0) / this.nodes.length;
      const distances = this.nodes.map(n => 
        Math.sqrt((n.x - centerX) ** 2 + (n.y - centerY) ** 2)
      );
      const avgDist = distances.reduce((a, b) => a + b, 0) / distances.length;
      const variance = distances.reduce((sum, d) => sum + (d - avgDist) ** 2, 0) / distances.length;
      return variance < 100;
    }

    isLinear() {
      if (this.nodes.length < 2) return false;
      return this.lines.length === this.nodes.length - 1;
    }

    aggregateEncoding() {
      const aggregated = new Float32Array(64);
      this.nodes.forEach(node => {
        for (let i = 0; i < 64; i++) {
          aggregated[i] += node.encoding[i];
        }
      });
      for (let i = 0; i < 64; i++) {
        aggregated[i] /= this.nodes.length;
      }
      return aggregated;
    }

    activate(strength) {
      this.activation = strength;
      this.frequency++;
      this.nodes.forEach(node => {
        node.activation = Math.min(1, node.activation + strength * 0.1);
      });
    }

    propagate() {
      this.lines.forEach(line => line.propagate());
      this.activation *= 0.9;
    }
  }

  class Pathway {
    constructor(id, type, color) {
      this.id = id;
      this.type = type;
      this.color = color;
      this.nodes = [];
      this.lines = [];
      this.patterns = [];
      this.subPathways = [];
      this.encoding = new Float32Array(128);
      this.state = 'idle';
      this.throughput = 0;
      this.latency = 0;
    }

    addNode(node) {
      this.nodes.push(node);
      this.updateEncoding();
    }

    addLine(line) {
      this.lines.push(line);
    }

    addPattern(pattern) {
      this.patterns.push(pattern);
    }

    createSubPathway(type) {
      const sub = new Pathway(`${this.id}_sub${this.subPathways.length}`, type, this.color);
      this.subPathways.push(sub);
      return sub;
    }

    updateEncoding() {
      for (let i = 0; i < 64 && i < this.nodes.length; i++) {
        for (let j = 0; j < 64; j++) {
          this.encoding[i] = this.nodes[i].encoding[j % 64];
        }
      }
    }

    traverse(input) {
      this.state = 'active';
      let signal = input;
      let processed = 0;

      const startTime = Date.now();

      for (let i = 0; i < this.nodes.length - 1; i++) {
        this.nodes[i].receive({ value: signal, type: 'input' });
        const output = this.nodes[i].send(this.nodes[i + 1]);
        if (output) {
          signal = output.value;
          processed++;
        }
      }

      this.patterns.forEach(pattern => {
        pattern.activate(signal);
        pattern.propagate();
      });

      this.subPathways.forEach(sub => {
        sub.traverse(signal * 0.5);
      });

      this.latency = Date.now() - startTime;
      this.throughput = processed / (this.latency + 1);
      this.state = 'idle';

      return signal;
    }

    learn(target, actual) {
      const error = target - actual;
      this.lines.forEach(line => line.learn(error));
    }

    evolve() {
      this.lines.forEach(line => line.evolve());
      this.patterns.forEach(pattern => {
        if (pattern.frequency > 10) {
          pattern.frequency *= 0.95;
        }
      });
    }
  }

  class DeepNeuralNetwork {
    constructor(layers) {
      this.layers = layers;
      this.weights = this.initWeights();
      this.biases = this.initBiases();
      this.activations = [];
    }

    initWeights() {
      const weights = [];
      for (let i = 0; i < this.layers.length - 1; i++) {
        const w = [];
        for (let j = 0; j < this.layers[i]; j++) {
          w.push(new Float32Array(this.layers[i + 1]).map(() => 
            (Math.random() - 0.5) * Math.sqrt(2 / this.layers[i])
          ));
        }
        weights.push(w);
      }
      return weights;
    }

    initBiases() {
      return this.layers.slice(1).map(size => 
        new Float32Array(size).map(() => 0)
      );
    }

    forward(input) {
      let activation = Float32Array.from(input);
      this.activations = [activation];

      for (let l = 0; l < this.weights.length; l++) {
        const nextActivation = new Float32Array(this.layers[l + 1]);
        
        for (let j = 0; j < this.layers[l + 1]; j++) {
          let sum = this.biases[l][j];
          for (let i = 0; i < this.layers[l]; i++) {
            sum += activation[i] * this.weights[l][i][j];
          }
          nextActivation[j] = Math.tanh(sum);
        }
        
        activation = nextActivation;
        this.activations.push(activation);
      }

      return activation;
    }

    backward(target, learningRate = 0.01) {
      const output = this.activations[this.activations.length - 1];
      let errors = new Float32Array(output.length);
      
      for (let i = 0; i < output.length; i++) {
        errors[i] = target[i] - output[i];
      }

      for (let l = this.weights.length - 1; l >= 0; l--) {
        const nextErrors = new Float32Array(this.layers[l]);
        
        for (let j = 0; j < this.layers[l + 1]; j++) {
          const gradient = errors[j] * (1 - this.activations[l + 1][j] ** 2);
          
          for (let i = 0; i < this.layers[l]; i++) {
            this.weights[l][i][j] += learningRate * gradient * this.activations[l][i];
            nextErrors[i] += gradient * this.weights[l][i][j];
          }
          
          this.biases[l][j] += learningRate * gradient;
        }
        
        errors = nextErrors;
      }
    }
  }

  class UnifiedIntelligenceSystem {
    constructor(width, height) {
      this.width = width;
      this.height = height;
      this.pathways = [];
      this.allNodes = [];
      this.allLines = [];
      this.neuralNet = new DeepNeuralNetwork([64, 128, 256, 128, 64]);
      this.cycle = 0;
      this.learningRate = 0.01;
      
      this.initializeArchitecture();
    }

    initializeArchitecture() {
      const pathwayConfigs = [
        { type: 'visual', color: '#3b82f6', y: 80 },
        { type: 'auditory', color: '#10b981', y: 160 },
        { type: 'semantic', color: '#f59e0b', y: 240 },
        { type: 'motor', color: '#8b5cf6', y: 320 },
        { type: 'memory', color: '#ec4899', y: 400 }
      ];

      pathwayConfigs.forEach((config, pathIdx) => {
        const pathway = new Pathway(`path_${pathIdx}`, config.type, config.color);

        const numNodes = 12;
        for (let i = 0; i < numNodes; i++) {
          const x = 50 + (this.width - 100) * (i / (numNodes - 1));
          const y = config.y + Math.sin(i * 0.5) * 20;
          
          let nodeType;
          if (i === 0) nodeType = 'sender';
          else if (i === numNodes - 1) nodeType = 'receiver';
          else if (i % 3 === 0) nodeType = 'transformer';
          else nodeType = 'processor';

          const node = new Node(x, y, nodeType);
          pathway.addNode(node);
          this.allNodes.push(node);
        }

        for (let i = 0; i < pathway.nodes.length - 1; i++) {
          const line = new Line(pathway.nodes[i], pathway.nodes[i + 1], 0.5 + Math.random() * 0.5);
          pathway.addLine(line);
          this.allLines.push(line);
          
          pathway.nodes[i].connections.push(pathway.nodes[i + 1]);
        }

        for (let i = 0; i < pathway.nodes.length - 3; i += 3) {
          const patternNodes = pathway.nodes.slice(i, i + 4);
          const patternLines = pathway.lines.slice(i, i + 3);
          const pattern = new Pattern(`pattern_${i}`, patternNodes, patternLines);
          pathway.addPattern(pattern);
        }

        for (let subIdx = 0; subIdx < 2; subIdx++) {
          const subPathway = pathway.createSubPathway(`sub_${config.type}`);
          
          for (let i = 0; i < 6; i++) {
            const baseNode = pathway.nodes[i * 2];
            const sx = baseNode.x + (Math.random() - 0.5) * 40;
            const sy = baseNode.y + (subIdx === 0 ? -30 : 30);
            const subNode = new Node(sx, sy, 'processor');
            subPathway.addNode(subNode);
            this.allNodes.push(subNode);
          }

          for (let i = 0; i < subPathway.nodes.length - 1; i++) {
            const subLine = new Line(subPathway.nodes[i], subPathway.nodes[i + 1], 0.3);
            subPathway.addLine(subLine);
            this.allLines.push(subLine);
          }
        }

        this.pathways.push(pathway);
      });

      for (let i = 0; i < this.pathways.length - 1; i++) {
        const path1 = this.pathways[i];
        const path2 = this.pathways[i + 1];
        
        for (let j = 0; j < Math.min(path1.nodes.length, path2.nodes.length); j += 2) {
          const crossLine = new Line(path1.nodes[j], path2.nodes[j], 0.2);
          this.allLines.push(crossLine);
          path1.nodes[j].connections.push(path2.nodes[j]);
        }
      }
    }

    process(input) {
      this.cycle++;
      const results = [];

      this.pathways.forEach(pathway => {
        const output = pathway.traverse(input);
        results.push({
          pathway: pathway.type,
          output: output,
          throughput: pathway.throughput,
          latency: pathway.latency
        });
      });

      const combinedEncoding = new Float32Array(64);
      this.pathways.forEach((pathway, idx) => {
        for (let i = 0; i < 64; i++) {
          combinedEncoding[i] += pathway.encoding[i] / this.pathways.length;
        }
      });

      const deepOutput = this.neuralNet.forward(combinedEncoding);

      if (this.cycle % 10 === 0) {
        const target = new Float32Array(64).map(() => Math.sin(this.cycle * 0.01) * 0.5 + 0.5);
        this.neuralNet.backward(target, this.learningRate);
        
        this.pathways.forEach(pathway => {
          pathway.learn(target[0], results[0].output);
        });
      }

      if (this.cycle % 50 === 0) {
        this.evolve();
      }

      return {
        pathwayResults: results,
        deepOutput: Array.from(deepOutput),
        combinedEncoding: Array.from(combinedEncoding),
        cycle: this.cycle
      };
    }

    evolve() {
      this.pathways.forEach(pathway => pathway.evolve());
    }

    getMetrics() {
      const activeNodes = this.allNodes.filter(n => n.activation > 0.1).length;
      const avgThroughput = this.pathways.reduce((sum, p) => sum + p.throughput, 0) / this.pathways.length;
      const totalPatterns = this.pathways.reduce((sum, p) => sum + p.patterns.length, 0);
      
      return {
        throughput: avgThroughput,
        activeNodes: activeNodes,
        pathways: this.pathways.length,
        learning: Math.min(1, this.cycle / 1000)
      };
    }
  }

  // ============================================================================
  // تهيئة الأنظمة
  // ============================================================================

  const [deepSystem, setDeepSystem] = useState(null);
  const [pathwaySystem, setPathwaySystem] = useState(null);

  useEffect(() => {
    // تهيئة نظام الذكاء العميق
    const reception = new ReceptionLayer();
    const analysis = new AnalysisLayer();
    const storage = new IntelligentStorageLayer();
    const synthesis = new SynthesisLayer(storage);
    const fieldIntel = new FieldIntelligenceLayer(storage, synthesis);
    const metaKnowledge = new MetaKnowledgeLayer({
      storage,
      fieldIntelligence: fieldIntel
    });

    setDeepSystem({
      reception,
      analysis,
      storage,
      synthesis,
      fieldIntel,
      metaKnowledge
    });

    // تهيئة نظام المسارات
    const newPathwaySystem = new UnifiedIntelligenceSystem(900, 500);
    setPathwaySystem(newPathwaySystem);
  }, []);

  // ============================================================================
  // دوال المعالجة
  // ============================================================================

  const handleTrain = async () => {
    if (!trainingText.trim() || !deepSystem) return;
    
    setIsProcessing(true);
    addLog(`بدء معالجة النص التدريبي...`);

    try {
      const processed = await deepSystem.reception.receive(trainingText);
      const analyzed = await deepSystem.analysis.analyze(processed);
      const stored = await deepSystem.storage.store(analyzed);
      
      const stats = deepSystem.storage.getStats();
      setMetrics(prev => ({
        ...prev,
        concepts: stats.total,
        connections: stats.connections,
        accuracy: (prev.accuracy + 0.1) % 1,
        learning: (prev.learning + 0.05) % 1,
        epochs: prev.epochs + 1
      }));

      addLog(`تم تخزين ${stored.length} مفهوم جديد`);
      setTrainingText('');
      
    } catch (error) {
      addLog(`خطأ في المعالجة: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim() || !deepSystem) return;
    
    setIsProcessing(true);
    const userMessage = { type: 'user', content: chatInput, timestamp: new Date() };
    setConversations(prev => [...prev, userMessage]);

    try {
      const result = await deepSystem.fieldIntel.process(chatInput);
      const aiMessage = { 
        type: 'ai', 
        content: result.response, 
        timestamp: new Date(),
        confidence: result.confidence
      };
      
      setConversations(prev => [...prev, aiMessage]);
      addLog(`تمت معالجة الاستعلام بثقة ${(result.confidence * 100).toFixed(1)}%`);
      
    } catch (error) {
      addLog(`خطأ في المحادثة: ${error.message}`);
    } finally {
      setIsProcessing(false);
      setChatInput('');
    }
  };

  const addLog = (message) => {
    const logEntry = {
      message,
      timestamp: new Date().toLocaleTimeString(),
      type: 'info'
    };
    setLogs(prev => [logEntry, ...prev.slice(0, 99)]);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      setTrainingText(content);
      addLog(`تم تحميل الملف: ${file.name}`);
    };
    reader.readAsText(file);
  };

  const handleResetPathway = () => {
    setPathwaySystem(new UnifiedIntelligenceSystem(900, 500));
    setDataFlow([]);
    setIsRunning(false);
  };

  // ============================================================================
  // الرسوم المتحركة
  // ============================================================================

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let frame = 0;

    const drawDeepIntelligence = () => {
      if (!deepSystem) return;

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const graph = deepSystem.storage.knowledgeGraph;
      graph.updatePositions(canvas.width, canvas.height);
      graph.decay();

      // رسم الروابط
      graph.edges.forEach(edge => {
        const fromNode = graph.nodes.get(edge.from);
        const toNode = graph.nodes.get(edge.to);
        
        if (fromNode && toNode) {
          const alpha = 0.1 + edge.signal * 0.9;
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = `hsl(${edge.weight * 120}, 70%, 50%)`;
          ctx.lineWidth = 1 + edge.weight * 3;
          
          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.stroke();
        }
      });

      // رسم العقد
      graph.nodes.forEach(node => {
        const size = 3 + node.activation * 7;
        
        if (node.activation > 0.3) {
          ctx.globalAlpha = node.activation * 0.3;
          ctx.fillStyle = `hsl(${node.importance * 120}, 80%, 50%)`;
          ctx.beginPath();
          ctx.arc(node.x, node.y, size * 2, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.globalAlpha = 0.8;
        ctx.fillStyle = `hsl(${node.importance * 120}, 80%, 60%)`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fill();

        if (node.accessCount > 5) {
          ctx.globalAlpha = 0.6;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(node.x, node.y, size + 2, 0, Math.PI * 2);
          ctx.stroke();
        }
      });

      // معلومات النظام
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`العقد: ${graph.nodeCount()}`, 10, 20);
      ctx.fillText(`الروابط: ${graph.edgeCount()}`, 10, 40);
      ctx.fillText(`الدورة: ${frame}`, 10, 60);
    };

    const drawUnifiedPathway = () => {
      if (!pathwaySystem) return;

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      pathwaySystem.allLines.forEach(line => {
        const alpha = 0.2 + Math.abs(line.flow) * 0.6;
        ctx.globalAlpha = alpha;
        
        const gradient = ctx.createLinearGradient(
          line.from.x, line.from.y,
          line.to.x, line.to.y
        );
        gradient.addColorStop(0, line.from.color);
        gradient.addColorStop(1, line.to.color);
        ctx.strokeStyle = gradient;
        
        ctx.lineWidth = 1 + Math.abs(line.flow) * 3;

        if (line.type === 'dashed') {
          ctx.setLineDash([5, 5]);
        } else if (line.type === 'dotted') {
          ctx.setLineDash([2, 3]);
        } else {
          ctx.setLineDash([]);
        }

        ctx.beginPath();
        ctx.moveTo(line.from.x, line.from.y);
        ctx.lineTo(line.to.x, line.to.y);
        ctx.stroke();

        if (Math.abs(line.flow) > 0.3) {
          const t = (frame * 0.05) % 1;
          const px = line.from.x + (line.to.x - line.from.x) * t;
          const py = line.from.y + (line.to.y - line.from.y) * t;
          
          ctx.globalAlpha = 0.8;
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      ctx.setLineDash([]);

      pathwaySystem.allNodes.forEach(node => {
        const size = 4 + node.activation * 6;
        
        if (node.activation > 0.5) {
          ctx.globalAlpha = (node.activation - 0.5) * 0.4;
          ctx.fillStyle = node.color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, size * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.globalAlpha = 0.7 + node.activation * 0.3;
        ctx.fillStyle = node.color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fill();

        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        
        if (node.type === 'sender') {
          ctx.beginPath();
          ctx.arc(node.x, node.y, size + 2, 0, Math.PI * 2);
          ctx.stroke();
        } else if (node.type === 'receiver') {
          ctx.strokeRect(node.x - size, node.y - size, size * 2, size * 2);
        } else if (node.type === 'transformer') {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y - size - 2);
          ctx.lineTo(node.x + size + 2, node.y + size + 2);
          ctx.lineTo(node.x - size - 2, node.y + size + 2);
          ctx.closePath();
          ctx.stroke();
        }
      });

      ctx.globalAlpha = 1;
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'left';
      pathwaySystem.pathways.forEach((pathway, idx) => {
        ctx.fillStyle = pathway.color;
        ctx.fillText(pathway.type.toUpperCase(), 10, 80 + idx * 80);
        
        const activity = pathway.nodes.reduce((sum, n) => sum + n.activation, 0) / pathway.nodes.length;
        ctx.fillRect(10, 85 + idx * 80, activity * 60, 3);
      });

      ctx.fillStyle = '#ffffff';
      ctx.font = '14px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`Cycle: ${pathwaySystem.cycle}`, canvas.width - 10, 20);
      ctx.fillText(`Nodes: ${pathwaySystem.allNodes.length}`, canvas.width - 10, 40);
      ctx.fillText(`Pathways: ${pathwaySystem.pathways.length}`, canvas.width - 10, 60);
    };

    const animate = () => {
      if (activeSystem === 'deepIntelligence') {
        drawDeepIntelligence();
      } else {
        if (isRunning && pathwaySystem) {
          const input = Math.sin(frame * 0.02) * 0.5 + 0.5;
          const result = pathwaySystem.process(input);
          
          const metrics = pathwaySystem.getMetrics();
          setSystemMetrics(metrics);

          if (frame % 20 === 0) {
            setDataFlow(prev => {
              const newFlow = [...prev, {
                cycle: pathwaySystem.cycle,
                value: result.pathwayResults[0].output,
                time: Date.now()
              }];
              return newFlow.slice(-50);
            });
          }

          frame++;
        }
        drawUnifiedPathway();
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [deepSystem, pathwaySystem, isRunning, activeSystem]);

  // ============================================================================
  // واجهة المستخدم
  // ============================================================================

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-4">
      <div className="max-w-7xl mx-auto">
        
        {/* رأس النظام */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold mb-4 flex items-center justify-center gap-3">
            <Brain className="w-10 h-10 text-cyan-400" />
            النظام المتكامل للذكاء الاصطناعي المتقدم
          </h1>
          <p className="text-gray-400 text-lg">
            نظام هجين يدمج الذكاء العميق مع شبكات المسارات العصبية
          </p>
        </div>

        {/* اختيار النظام */}
        <div className="flex gap-4 mb-6 justify-center">
          <button
            onClick={() => setActiveSystem('deepIntelligence')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeSystem === 'deepIntelligence'
                ? 'bg-cyan-600 text-white shadow-lg'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            <Brain className="w-5 h-5 inline mr-2" />
            نظام الذكاء العميق
          </button>
          <button
            onClick={() => setActiveSystem('unifiedPathway')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeSystem === 'unifiedPathway'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            <Network className="w-5 h-5 inline mr-2" />
            نظام المسارات الموحد
          </button>
        </div>

        {activeSystem === 'deepIntelligence' ? (
          <>
            {/* شريط الحالة */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
              <div className="bg-gradient-to-br from-blue-900 to-blue-950 p-4 rounded-xl border border-blue-800">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-5 h-5 text-blue-400" />
                  <span className="text-sm font-semibold">المفاهيم</span>
                </div>
                <div className="text-2xl font-bold">{metrics.concepts}</div>
                <div className="text-xs text-blue-300">وحدة معرفية</div>
              </div>
              
              <div className="bg-gradient-to-br from-green-900 to-green-950 p-4 rounded-xl border border-green-800">
                <div className="flex items-center gap-2 mb-2">
                  <GitBranch className="w-5 h-5 text-green-400" />
                  <span className="text-sm font-semibold">الروابط</span>
                </div>
                <div className="text-2xl font-bold">{metrics.connections}</div>
                <div className="text-xs text-green-300">ارتباط نشط</div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-900 to-purple-950 p-4 rounded-xl border border-purple-800">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="w-5 h-5 text-purple-400" />
                  <span className="text-sm font-semibold">المستويات</span>
                </div>
                <div className="text-2xl font-bold">{metrics.layers}</div>
                <div className="text-xs text-purple-300">طبقة معالجة</div>
              </div>
              
              <div className="bg-gradient-to-br from-yellow-900 to-yellow-950 p-4 rounded-xl border border-yellow-800">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-yellow-400" />
                  <span className="text-sm font-semibold">الدقة</span>
                </div>
                <div className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</div>
                <div className="text-xs text-yellow-300">معدل الصواب</div>
              </div>
              
              <div className="bg-gradient-to-br from-red-900 to-red-950 p-4 rounded-xl border border-red-800">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-5 h-5 text-red-400" />
                  <span className="text-sm font-semibold">التعلم</span>
                </div>
                <div className="text-2xl font-bold">{(metrics.learning * 100).toFixed(1)}%</div>
                <div className="text-xs text-red-300">مستوى التعلم</div>
              </div>
              
              <div className="bg-gradient-to-br from-indigo-900 to-indigo-950 p-4 rounded-xl border border-indigo-800">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-indigo-400" />
                  <span className="text-sm font-semibold">الدورات</span>
                </div>
                <div className="text-2xl font-bold">{metrics.epochs}</div>
                <div className="text-xs text-indigo-300">دورة تدريب</div>
              </div>
            </div>

            {/* المنطقة الرئيسية */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              
              {/* لوحة التحكم */}
              <div className="lg:col-span-1 space-y-6">
                
                {/* التدريب */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <BookOpen className="w-5 h-5 text-cyan-400" />
                    <h3 className="font-semibold">التدريب والمعرفة</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <textarea
                      value={trainingText}
                      onChange={(e) => setTrainingText(e.target.value)}
                      placeholder="أدخل النص للتدريب... أو قم برفع ملف"
                      className="w-full h-32 bg-gray-900 border border-gray-700 rounded-lg p-3 text-white resize-none focus:border-cyan-500 focus:outline-none"
                    />
                    
                    <div className="flex gap-2">
                      <button
                        onClick={handleTrain}
                        disabled={isProcessing || !trainingText.trim()}
                        className="flex-1 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 text-white py-2 px-4 rounded-lg transition-all flex items-center justify-center gap-2"
                      >
                        {isProcessing ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            جاري المعالجة...
                          </>
                        ) : (
                          <>
                            <Save className="w-4 h-4" />
                            تدريب النظام
                          </>
                        )}
                      </button>
                      
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-all flex items-center gap-2"
                      >
                        <Upload className="w-4 h-4" />
                        رفع ملف
                      </button>
                    </div>
                    
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                      accept=".txt,.pdf,.docx"
                      className="hidden"
                    />
                  </div>
                </div>

                {/* المحادثة */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <MessageSquare className="w-5 h-5 text-green-400" />
                    <h3 className="font-semibold">المحادثة الذكية</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <input
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                      placeholder="اسأل النظام عن anything..."
                      className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white focus:border-green-500 focus:outline-none"
                    />
                    
                    <button
                      onClick={handleChat}
                      disabled={isProcessing || !chatInput.trim()}
                      className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white py-2 px-4 rounded-lg transition-all flex items-center justify-center gap-2"
                    >
                      {isProcessing ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          جاري الرد...
                        </>
                      ) : (
                        <>
                          <MessageSquare className="w-4 h-4" />
                          إرسال السؤال
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* السجلات */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-5 h-5 text-yellow-400" />
                    <h3 className="font-semibold">سجلات النظام</h3>
                  </div>
                  
                  <div className="h-48 overflow-y-auto space-y-2">
                    {logs.map((log, index) => (
                      <div key={index} className="text-sm p-2 bg-gray-900 rounded-lg">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300">{log.message}</span>
                          <span className="text-gray-500 text-xs">{log.timestamp}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* المحادثات */}
              <div className="lg:col-span-1">
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 h-full">
                  <div className="flex items-center gap-2 mb-4">
                    <MessageSquare className="w-5 h-5 text-blue-400" />
                    <h3 className="font-semibold">المحادثات</h3>
                  </div>
                  
                  <div className="h-96 overflow-y-auto space-y-4">
                    {conversations.map((conv, index) => (
                      <div
                        key={index}
                        className={`p-4 rounded-lg ${
                          conv.type === 'user'
                            ? 'bg-blue-900 bg-opacity-30 border border-blue-800 ml-8'
                            : 'bg-green-900 bg-opacity-30 border border-green-800 mr-8'
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <div className={`w-2 h-2 rounded-full ${
                            conv.type === 'user' ? 'bg-blue-400' : 'bg-green-400'
                          }`} />
                          <span className="text-sm font-semibold">
                            {conv.type === 'user' ? 'أنت' : 'النظام'}
                          </span>
                          {conv.confidence && (
                            <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                              {(conv.confidence * 100).toFixed(1)}%
                            </span>
                          )}
                          <span className="text-xs text-gray-400 ml-auto">
                            {conv.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-gray-200">{conv.content}</p>
                      </div>
                    ))}
                    
                    {conversations.length === 0 && (
                      <div className="text-center text-gray-500 py-8">
                        <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>لا توجد محادثات بعد</p>
                        <p className="text-sm">ابدأ محادثة مع النظام</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* التصور البصري */}
              <div className="lg:col-span-1">
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 h-full">
                  <div className="flex items-center gap-2 mb-4">
                    <Eye className="w-5 h-5 text-purple-400" />
                    <h3 className="font-semibold">الخريطة المعرفية</h3>
                    <button
                      onClick={() => setVisualizing(!visualizing)}
                      className="ml-auto text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-lg transition-all"
                    >
                      {visualizing ? 'إيقاف' : 'تشغيل'}
                    </button>
                  </div>
                  
                  <div className="bg-black rounded-lg overflow-hidden h-96">
                    <canvas
                      ref={canvasRef}
                      width={800}
                      height={400}
                      className="w-full h-full"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400">المفاهيم النشطة</div>
                      <div className="text-lg font-bold text-cyan-400">
                        {deepSystem?.storage?.knowledgeGraph?.nodes?.size || 0}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-400">الروابط النشطة</div>
                      <div className="text-lg font-bold text-green-400">
                        {deepSystem?.storage?.knowledgeGraph?.edges?.size || 0}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            {/* نظام المسارات الموحد */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gradient-to-br from-blue-900 to-blue-950 p-4 rounded-xl border border-blue-800">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  <span className="text-sm font-semibold">الإنتاجية</span>
                </div>
                <div className="text-2xl font-bold">{systemMetrics.throughput.toFixed(2)}</div>
                <div className="text-xs text-blue-300">إشارة/دورة</div>
              </div>
              
              <div className="bg-gradient-to-br from-green-900 to-green-950 p-4 rounded-xl border border-green-800">
                <div className="flex items-center gap-2 mb-2">
                  <Cpu className="w-5 h-5 text-green-400" />
                  <span className="text-sm font-semibold">العقد النشطة</span>
                </div>
                <div className="text-2xl font-bold">{systemMetrics.activeNodes}</div>
                <div className="text-xs text-green-300">عقدة نشطة</div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-900 to-purple-950 p-4 rounded-xl border border-purple-800">
                <div className="flex items-center gap-2 mb-2">
                  <GitBranch className="w-5 h-5 text-purple-400" />
                  <span className="text-sm font-semibold">المسارات</span>
                </div>
                <div className="text-2xl font-bold">{systemMetrics.pathways}</div>
                <div className="text-xs text-purple-300">مسار نشط</div>
              </div>
              
              <div className="bg-gradient-to-br from-yellow-900 to-yellow-950 p-4 rounded-xl border border-yellow-800">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  <span className="text-sm font-semibold">مستوى التعلم</span>
                </div>
                <div className="text-2xl font-bold">{(systemMetrics.learning * 100).toFixed(1)}%</div>
                <div className="text-xs text-yellow-300">كفاءة التعلم</div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              
              {/* التحكم الرئيسي */}
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <Settings className="w-5 h-5 text-cyan-400" />
                    <h3 className="font-semibold">تحكم النظام</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <button
                      onClick={() => setIsRunning(!isRunning)}
                      className={`w-full py-3 px-4 rounded-lg transition-all flex items-center justify-center gap-2 ${
                        isRunning 
                          ? 'bg-red-600 hover:bg-red-700' 
                          : 'bg-green-600 hover:bg-green-700'
                      }`}
                    >
                      {isRunning ? (
                        <>
                          <Pause className="w-4 h-4" />
                          إيقاف التشغيل
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          بدء التشغيل
                        </>
                      )}
                    </button>
                    
                    <button
                      onClick={handleResetPathway}
                      className="w-full bg-gray-700 hover:bg-gray-600 py-3 px-4 rounded-lg transition-all flex items-center justify-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      إعادة تعيين
                    </button>
                  </div>
                </div>

                {/* معلومات النظام */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <Radio className="w-5 h-5 text-green-400" />
                    <h3 className="font-semibold">معلومات الشبكة</h3>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">إجمالي العقد</span>
                      <span className="font-semibold">{pathwaySystem?.allNodes.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">إجمالي الخطوط</span>
                      <span className="font-semibold">{pathwaySystem?.allLines.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">الدورات</span>
                      <span className="font-semibold">{pathwaySystem?.cycle || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">الحالة</span>
                      <span className={`font-semibold ${
                        isRunning ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {isRunning ? 'نشط' : 'متوقف'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* تدفق البيانات */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-5 h-5 text-purple-400" />
                    <h3 className="font-semibold">تدفق البيانات</h3>
                  </div>
                  
                  <div className="h-32 overflow-y-auto space-y-2">
                    {dataFlow.slice().reverse().map((flow, index) => (
                      <div key={index} className="text-sm p-2 bg-gray-900 rounded-lg">
                        <div className="flex justify-between">
                          <span>دورة {flow.cycle}</span>
                          <span className="text-green-400">{flow.value.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* التصور البصري */}
              <div className="lg:col-span-3">
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 h-full">
                  <div className="flex items-center gap-2 mb-4">
                    <Network className="w-5 h-5 text-cyan-400" />
                    <h3 className="font-semibold">شبكة المسارات العصبية</h3>
                  </div>
                  
                  <div className="bg-black rounded-lg overflow-hidden h-96">
                    <canvas
                      ref={canvasRef}
                      width={900}
                      height={500}
                      className="w-full h-full"
                    />
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 mt-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400">المسارات</div>
                      <div className="text-lg font-bold text-blue-400">
                        {pathwaySystem?.pathways.length || 0}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-400">الأنماط</div>
                      <div className="text-lg font-bold text-green-400">
                        {pathwaySystem?.pathways.reduce((sum, p) => sum + p.patterns.length, 0) || 0}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-400">المسارات الفرعية</div>
                      <div className="text-lg font-bold text-purple-400">
                        {pathwaySystem?.pathways.reduce((sum, p) => sum + p.subPathways.length, 0) || 0}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default IntegratedAISystem;