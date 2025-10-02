import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import logging
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline.benchmark import run_benchmark, get_dataset_path, get_dataset_loader

st.set_page_config(layout="wide", page_title="ETSA Benchmark UI")

if 'selected_run' not in st.session_state:
    st.session_state.selected_run = None
if 'show_batch_comparison' not in st.session_state:
    st.session_state.show_batch_comparison = False
if 'batch_runs' not in st.session_state:
    st.session_state.batch_runs = []

st.title("ETSA Benchmark and Analysis UI")

BASE_OUTPUT_DIR = os.path.join("outputs", "benchmarks")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def get_run_dirs():
    if not os.path.exists(BASE_OUTPUT_DIR):
        return []
    try:
        run_dirs = [d for d in os.listdir(BASE_OUTPUT_DIR) if os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d))]
        return sorted(run_dirs, reverse=True)
    except FileNotFoundError:
        return []

def execute_benchmark_run(dataset: str, mode: str, limit: int, pos_thresh: float, neg_thresh: float):
    if mode == "run_all_modes":
        all_modes = ["full_stack", "efficiency", "no_formulas", "vader_baseline", "transformer_absa", "ner_basic", "no_modifiers", "no_relations"]
        batch_name = f"{dataset}_all_modes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        st.info(f"Starting batch benchmark run: **{batch_name}** ({len(all_modes)} modes)")
        
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        completed_runs = []
        prev_dirs = set(get_run_dirs())
        for i, current_mode in enumerate(all_modes):
            overall_status.text(f"Running mode {i+1}/{len(all_modes)}: {current_mode}")
            run_name = f"{selected_dataset}_{selected_mode}"                           
            try:
                with st.expander(f"Running {current_mode}...", expanded=False):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def progress_callback(current, total):
                        if total > 0:
                            progress = min(1.0, (current + 1) / total)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing item {current + 1}/{total}...")
                    run_benchmark(
                        run_name=run_name,
                        dataset_name=dataset,
                        run_mode=current_mode,
                        limit=limit,
                        pos_thresh=pos_thresh,
                        neg_thresh=neg_thresh,
                        progress_callback=progress_callback
                    )
                    st.success(f"✅ {current_mode} completed")
                                                    
                    new_dirs = set(get_run_dirs()) - prev_dirs
                    if new_dirs:
                                                    
                        completed_runs.append(sorted(new_dirs)[-1])
                        prev_dirs = set(get_run_dirs())
            except Exception as e:
                st.error(f"❌ {current_mode} failed: {e}")
            overall_progress.progress((i + 1) / len(all_modes))
        overall_status.text(f"Batch completed! {len(completed_runs)}/{len(all_modes)} modes successful")
        st.success(f"Batch benchmark '{batch_name}' completed! {len(completed_runs)} successful runs.")
        if completed_runs:
            st.session_state.selected_run = completed_runs[0]
            st.session_state.batch_runs = completed_runs
            st.session_state.show_batch_comparison = True
        
    else:
        run_name = f"{selected_dataset}_{selected_mode}"                           
        
        st.info(f"Starting benchmark run: **{run_name}**")

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("Benchmark Run Logs", expanded=True)
        log_placeholder = log_container.empty()
        log_output = ""

        def progress_callback(current, total):
            if total > 0:
                progress = min(1.0, (current + 1) / total)
                progress_bar.progress(progress)
                status_text.text(f"Processing item {current + 1}/{total}...")
        
        log_stream = ""
        class StreamlitLogHandler(logging.Handler):
            def emit(self, record):
                nonlocal log_output
                msg = self.format(record)
                log_output += msg + "\n"
                log_placeholder.code(log_output, language='log')

        benchmark_logger = logging.getLogger('src.pipeline.benchmark')
        benchmark_logger.setLevel(logging.INFO)
        if benchmark_logger.hasHandlers():
            benchmark_logger.handlers.clear()
        benchmark_logger.addHandler(StreamlitLogHandler())

        try:
            prev_dirs = set(get_run_dirs())
            run_benchmark(
                run_name=run_name,
                dataset_name=dataset,
                run_mode=mode,
                limit=limit,
                pos_thresh=pos_thresh,
                neg_thresh=neg_thresh,
                progress_callback=progress_callback
            )
            st.success(f"Benchmark '{run_name}' completed successfully!")
            status_text.text("Completed!")
                                            
            new_dirs = set(get_run_dirs()) - prev_dirs
            if new_dirs:
                actual_run_name = sorted(new_dirs)[-1]
                st.session_state.selected_run = actual_run_name
            else:
                st.session_state.selected_run = run_name
        except Exception as e:
            st.error(f"Benchmark '{run_name}' failed with an exception: {e}")
            status_text.text(f"Failed! Check logs for details.")
            logging.error(f"Exception during benchmark run: {e}", exc_info=True)

    st.rerun()


def get_run_comparison_data():
    run_dirs = get_run_dirs()
    comparison_data = []
    
    for run_dir in run_dirs:
        metrics_path = os.path.join(BASE_OUTPUT_DIR, run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                parts = run_dir.split('_')
                if len(parts) >= 4:
                    if 'test' in parts[0] and 'laptop' in parts[1] or 'restaurant' in parts[1]:
                        dataset = '_'.join(parts[:3])
                        mode_parts = []
                        timestamp_parts = []
                        
                        for i, part in enumerate(parts[3:], 3):
                            if part.isdigit() and len(part) == 8:
                                timestamp_parts = parts[i:]
                                break
                            mode_parts.append(part)
                        
                        mode = '_'.join(mode_parts) if mode_parts else 'unknown'
                        timestamp = '_'.join(timestamp_parts) if timestamp_parts else 'unknown'
                    else:
                        dataset = parts[0] if parts else "unknown"
                        mode = parts[1] if len(parts) > 1 else "unknown"
                        timestamp = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"
                else:
                    dataset = "unknown"
                    mode = "unknown"
                    timestamp = run_dir
                
                comparison_data.append({
                    'run_name': run_dir,
                    'dataset': dataset,
                    'mode': mode,
                    'timestamp': timestamp,
                    'accuracy': metrics.get('accuracy', 0),
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                    'macro_f1': metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0),
                    'weighted_f1': metrics.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0)
                })
            except Exception as e:
                continue
    
    return pd.DataFrame(comparison_data)

def display_batch_comparison():
    st.header("🚀 Batch Run Analysis - Just Completed!")
    
    if 'batch_runs' not in st.session_state or not st.session_state.batch_runs:
        st.error("No batch runs found!")
        return
    
    batch_runs = st.session_state.batch_runs
    st.info(f"Analyzing {len(batch_runs)} runs that just completed")
    
    batch_data = []
    for run_name in batch_runs:
        metrics_path = os.path.join(BASE_OUTPUT_DIR, run_name, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                parts = run_name.split('_')
                                           
                timestamp_idx = None
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():
                        timestamp_idx = i
                        break
                
                                                                              
                known_modes = ['transformer_absa', 'vader_baseline', 'full_stack', 'ner_basic', 'no_modifiers', 'no_relations', 'baseline', 'absa']
                
                if timestamp_idx:
                    pre_timestamp = '_'.join(parts[:timestamp_idx])
                    
                                                                     
                    cleaned_pre_timestamp = pre_timestamp
                    if cleaned_pre_timestamp.startswith('single_run_'):
                        cleaned_pre_timestamp = cleaned_pre_timestamp[11:]                        
                    elif cleaned_pre_timestamp.startswith('batch_') and '_' in cleaned_pre_timestamp[6:]:
                                                   
                        first_underscore_after_batch = cleaned_pre_timestamp.find('_', 6)
                        if first_underscore_after_batch != -1:
                            cleaned_pre_timestamp = cleaned_pre_timestamp[first_underscore_after_batch + 1:]
                    
                                                             
                    mode = 'unknown'
                    dataset = 'unknown'
                    
                    for known_mode in known_modes:
                        if known_mode in cleaned_pre_timestamp:
                            mode = known_mode
                                                                   
                            mode_start = cleaned_pre_timestamp.find(known_mode)
                            if mode_start > 0:
                                dataset = cleaned_pre_timestamp[:mode_start].rstrip('_')
                            else:
                                                                                 
                                dataset = "unknown"
                            break
                    
                    if mode == 'unknown':
                                                                              
                        cleaned_parts = cleaned_pre_timestamp.split('_')
                        if len(cleaned_parts) >= 2:
                            dataset = '_'.join(cleaned_parts[:-1])
                            mode = cleaned_parts[-1]
                        else:
                            dataset = "unknown"
                            mode = cleaned_parts[0] if cleaned_parts else "unknown"
                else:
                    dataset = "unknown"
                    mode = "unknown"
                
                batch_data.append({
                    'run_name': run_name,
                    'mode': mode,
                    'dataset': dataset,
                    'accuracy': metrics.get('accuracy', 0),
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                    'macro_f1': metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0),
                })
            except Exception as e:
                continue
    
    if not batch_data:
        st.error("Could not load batch run data!")
        return
        
    df = pd.DataFrame(batch_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Performance Summary")
        best_acc = df.loc[df['accuracy'].idxmax()]
        best_f1 = df.loc[df['macro_f1'].idxmax()]
        
        st.metric("🥇 Best Accuracy", f"{best_acc['accuracy']:.2%}", help=f"Mode: {best_acc['mode']}")
        st.metric("🎯 Best F1-Score", f"{best_f1['macro_f1']:.3f}", help=f"Mode: {best_f1['mode']}")
        
        st.markdown("**🔍 Quick Insights:**")
        if 'full_stack' in df['mode'].values:
            full_stack_acc = df[df['mode'] == 'full_stack']['accuracy'].iloc[0]
            st.write(f"• Full stack baseline: {full_stack_acc:.2%}")
            
            for mode in ['no_modifiers', 'no_relations', 'ner_basic']:
                if mode in df['mode'].values:
                    mode_acc = df[df['mode'] == mode]['accuracy'].iloc[0]
                    impact = full_stack_acc - mode_acc
                    st.write(f"• {mode.replace('_', ' ').title()}: {mode_acc:.2%} (Δ {impact:+.1%})")
    
    with col2:
        st.subheader("📊 Mode Comparison")
        display_df = df[['mode', 'accuracy', 'macro_f1']].copy()
        display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2%}")
        display_df['macro_f1'] = display_df['macro_f1'].apply(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True)
    
    st.subheader("📈 Performance Visualization")
    
    import plotly.express as px
    
    fig = px.bar(
        df, 
        x='mode', 
        y='accuracy',
        title="Accuracy Comparison Across All Modes",
        labels={'accuracy': 'Accuracy', 'mode': 'Pipeline Mode'},
        color='accuracy',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(yaxis_tickformat='.1%', xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🔄 Clear Batch Results", use_container_width=True):
        st.session_state.show_batch_comparison = False
        st.session_state.batch_runs = []
        st.rerun()


def display_comparison_analysis():
    st.header("📊 Multi-Run Comparison Analysis")
    
    df = get_run_comparison_data()
    
    if df.empty:
        st.info("No benchmark runs found for comparison. Run some benchmarks first!")
        return
    
    st.subheader("📋 All Runs Overview")
    display_df = df[['run_name', 'dataset', 'mode', 'accuracy', 'balanced_accuracy', 'macro_f1']].copy()
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2%}")
    display_df['balanced_accuracy'] = display_df['balanced_accuracy'].apply(lambda x: f"{x:.2%}")
    display_df['macro_f1'] = display_df['macro_f1'].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True)
    
    if len(df['dataset'].unique()) > 1:
        st.subheader("🔬 Dataset Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_datasets = st.multiselect("Select Datasets", options=df['dataset'].unique(), default=df['dataset'].unique())
        with col2:
            selected_modes = st.multiselect("Select Modes", options=df['mode'].unique(), default=df['mode'].unique())
        
        filtered_df = df[(df['dataset'].isin(selected_datasets)) & (df['mode'].isin(selected_modes))]
        
        if not filtered_df.empty:
            tab1, tab2, tab3 = st.tabs(["📈 Performance Charts", "🏆 Best Results", "📋 Detailed Comparison"])
            
            with tab1:
                st.markdown("#### Accuracy by Mode")
                
                import plotly.express as px
                
                fig_acc = px.bar(
                    filtered_df, 
                    x='mode', 
                    y='accuracy', 
                    color='dataset',
                    title="Accuracy Comparison by Mode",
                    labels={'accuracy': 'Accuracy', 'mode': 'Pipeline Mode'}
                )
                fig_acc.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_acc, use_container_width=True)
                
                st.markdown("#### F1-Score Comparison")
                fig_f1 = px.bar(
                    filtered_df, 
                    x='mode', 
                    y='macro_f1', 
                    color='dataset',
                    title="Macro F1-Score by Mode",
                    labels={'macro_f1': 'Macro F1-Score', 'mode': 'Pipeline Mode'}
                )
                st.plotly_chart(fig_f1, use_container_width=True)
            
            with tab2:
                st.markdown("#### 🥇 Top Performers")
                
                best_overall = filtered_df.loc[filtered_df['accuracy'].idxmax()]
                best_f1 = filtered_df.loc[filtered_df['macro_f1'].idxmax()]
                best_balanced = filtered_df.loc[filtered_df['balanced_accuracy'].idxmax()]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🎯 Best Accuracy", 
                        f"{best_overall['accuracy']:.2%}",
                        help=f"Mode: {best_overall['mode']}\nDataset: {best_overall['dataset']}\nRun: {best_overall['run_name']}"
                    )
                    st.caption(f"**{best_overall['mode']}** on {best_overall['dataset']}")
                
                with col2:
                    st.metric(
                        "🎪 Best F1-Score", 
                        f"{best_f1['macro_f1']:.3f}",
                        help=f"Mode: {best_f1['mode']}\nDataset: {best_f1['dataset']}\nRun: {best_f1['run_name']}"
                    )
                    st.caption(f"**{best_f1['mode']}** on {best_f1['dataset']}")
                
                with col3:
                    st.metric(
                        "⚖️ Best Balanced Acc", 
                        f"{best_balanced['balanced_accuracy']:.2%}",
                        help=f"Mode: {best_balanced['mode']}\nDataset: {best_balanced['dataset']}\nRun: {best_balanced['run_name']}"
                    )
                    st.caption(f"**{best_balanced['mode']}** on {best_balanced['dataset']}")
                
                st.markdown("#### 📊 Mode Performance Summary")
                mode_summary = filtered_df.groupby('mode').agg({
                    'accuracy': ['mean', 'std', 'max'],
                    'macro_f1': ['mean', 'std', 'max'],
                    'balanced_accuracy': ['mean', 'std', 'max']
                }).round(4)
                
                mode_summary.columns = ['_'.join(col).strip() for col in mode_summary.columns]
                st.dataframe(mode_summary, use_container_width=True)
            
            with tab3:
                st.markdown("#### 🔍 Detailed Performance Matrix")
                
                pivot_acc = filtered_df.pivot_table(
                    values='accuracy', 
                    index='mode', 
                    columns='dataset', 
                    aggfunc='mean'
                ).fillna(0)
                
                st.markdown("**Accuracy by Mode and Dataset:**")
                styled_pivot = pivot_acc.style.format("{:.2%}").background_gradient(cmap='RdYlGn', vmin=0, vmax=1)
                st.dataframe(styled_pivot, use_container_width=True)
                
                pivot_f1 = filtered_df.pivot_table(
                    values='macro_f1', 
                    index='mode', 
                    columns='dataset', 
                    aggfunc='mean'
                ).fillna(0)
                
                st.markdown("**Macro F1-Score by Mode and Dataset:**")
                styled_pivot_f1 = pivot_f1.style.format("{:.3f}").background_gradient(cmap='RdYlGn', vmin=0, vmax=1)
                st.dataframe(styled_pivot_f1, use_container_width=True)
                
                st.markdown("#### 🎭 Ablation Study Insights")
                if 'full_stack' in filtered_df['mode'].values:
                    full_stack_results = filtered_df[filtered_df['mode'] == 'full_stack']
                    if not full_stack_results.empty:
                        baseline_acc = full_stack_results['accuracy'].mean()
                        
                        ablation_comparison = []
                        for mode in ['no_modifiers', 'no_relations', 'ner_basic']:
                            mode_results = filtered_df[filtered_df['mode'] == mode]
                            if not mode_results.empty:
                                mode_acc = mode_results['accuracy'].mean()
                                impact = baseline_acc - mode_acc
                                ablation_comparison.append({
                                    'Component Removed': mode.replace('_', ' ').title(),
                                    'Accuracy Drop': f"{impact:.2%}",
                                    'Relative Impact': f"{impact/baseline_acc:.1%}" if baseline_acc > 0 else "N/A"
                                })
                        
                        if ablation_comparison:
                            ablation_df = pd.DataFrame(ablation_comparison)
                            st.dataframe(ablation_df, use_container_width=True)
                        else:
                            st.info("Run ablation modes (no_modifiers, no_relations, ner_basic) to see component impact analysis.")
                    else:
                        st.info("Run the 'full_stack' mode to enable ablation study insights.")
        else:
            st.warning("No data matches the selected filters.")
    else:
        st.info("Multiple datasets needed for comparison analysis. Run benchmarks on different datasets!")


def display_results(run_name: str):
    if not run_name:
        st.info("Select a benchmark run from the sidebar to view its results.")
        return

    run_dir = os.path.join(BASE_OUTPUT_DIR, run_name)
    if not os.path.exists(run_dir):
        st.error(f"Directory for run '{run_name}' not found.")
        return

    st.header(f"Results for: {run_name}")

    metrics_path = os.path.join(run_dir, "metrics.json")
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    errors_path = os.path.join(run_dir, "error_analysis.csv")
    log_path = os.path.join(run_dir, "benchmark.log")

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    df_errors = pd.DataFrame()
    if os.path.exists(errors_path):
        df_errors = pd.read_csv(errors_path)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "🧮 Confusion Matrix", "🐞 Error Analysis", "📝 Logs", "🔬 Comparison"])

    with tab1:
        if metrics:
            st.subheader("Overall Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            col2.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.2%}")

            st.subheader("Classification Report")
            report = metrics.get("classification_report", {})
            if report:
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report)
            else:
                st.warning("Classification report not found in metrics file.")
        else:
            st.warning("Metrics file (metrics.json) not found.")

    with tab2:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.warning("Confusion matrix image (confusion_matrix.png) not found.")

    with tab3:
        st.subheader("🔍 Interactive Error Analysis")
        if not df_errors.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Errors", len(df_errors))
            col2.metric("Unique Items with Errors", df_errors['id'].nunique())
            if len(df_errors) > 0:
                most_common = df_errors.groupby(['gold', 'pred']).size().idxmax()
                most_common_str = f"{most_common[0]} → {most_common[1]}"
            else:
                most_common_str = "N/A"
            col3.metric("Most Common Error", most_common_str)
            
            st.markdown("#### Filter Errors")
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                gold_filter = st.multiselect("Gold Label", options=df_errors['gold'].unique(), default=df_errors['gold'].unique())
            with filter_col2:
                pred_filter = st.multiselect("Predicted Label", options=df_errors['pred'].unique(), default=df_errors['pred'].unique())
            
            filtered_df = df_errors[
                (df_errors['gold'].isin(gold_filter)) & 
                (df_errors['pred'].isin(pred_filter))
            ]
            
            st.markdown("#### Error Details")
            st.write("Click on a row to see detailed pipeline trace and module analysis.")
            
            if 'selected_error_index' not in st.session_state:
                st.session_state.selected_error_index = None

            for idx, row in filtered_df.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
                    
                    if col1.button("🔍", key=f"select_{idx}"):
                        st.session_state.selected_error_index = idx
                    
                    col2.write(f"**{row['term']}**: {row['text'][:100]}...")
                    col3.write(f"Gold: **{row['gold']}**")
                    col4.write(f"Pred: **{row['pred']}**")
                    col5.write(f"Score: {row['score']:.3f}")

            if st.session_state.selected_error_index is not None:
                selected_row = df_errors.loc[st.session_state.selected_error_index]
                trace_path = selected_row.get('trace_path')
                
                if trace_path and os.path.exists(trace_path):
                    with open(trace_path, 'r') as f:
                        trace_data = json.load(f)
                    
                    st.markdown("---")
                    st.markdown(f"### 🔬 Deep Dive Analysis: Item {selected_row['id']}")
                    
                    analysis_tabs = st.tabs([
                        "📋 Executive Summary", 
                        "🔄 Pipeline Trace", 
                        "🧩 Module Analysis", 
                        "📊 Graph Visualization",
                        "⚙️ Parameter Testing",
                        "📥 Export Trace"
                    ])
                    
                    with analysis_tabs[0]:                     
                        st.markdown("#### 🎯 Error Summary")
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.markdown(f"**Item ID:** {trace_data['item_id']}")
                            st.markdown(f"**Pipeline Mode:** {trace_data['pipeline_mode']}")
                            st.markdown(f"**Aspect Term:** {selected_row['term']}")
                            st.markdown(f"**Expected:** {selected_row['gold']}")
                            st.markdown(f"**Predicted:** {selected_row['pred']}")
                            st.markdown(f"**Confidence Score:** {selected_row['score']:.3f}")
                        
                        with summary_col2:
                            st.markdown("**Input Text:**")
                            st.text_area("Input Text", trace_data['input_text'], height=100, disabled=True, key="summary_text", label_visibility="hidden")
                        
                        exec_trace = trace_data.get('execution_trace', {})
                        actual_exec_trace = exec_trace.get('execution_trace', {})
                        modules = actual_exec_trace.get('modules', {})
                        
                        st.markdown("#### 🏗️ Module Performance Overview")
                        if modules:
                            main_modules = {k: v for k, v in modules.items() if not k.endswith('_errors')}
                            if main_modules:
                                module_cols = st.columns(len(main_modules))
                                
                                for i, (module_name, module_data) in enumerate(main_modules.items()):
                                    status = module_data.get('status', 'unknown')
                                    status_emoji = "✅" if status == "success" else "⚠️" if status == "fallback" else "❌"
                                    
                                    with module_cols[i]:
                                        st.metric(
                                            f"{status_emoji} {module_name.replace('_', ' ').title()}",
                                            status.title(),
                                            help=f"Module: {module_name}\nStatus: {status}"
                                        )
                            else:
                                st.info("No main module data available for this trace.")
                        else:
                            st.info("No module execution data available for this trace.")
                    
                    with analysis_tabs[1]:                  
                        st.markdown("#### 🔄 Step-by-Step Pipeline Execution")
                        
                        actual_exec_trace = exec_trace.get('execution_trace', {})
                        stages = actual_exec_trace.get('stages', [])
                        if stages:
                            for i, stage in enumerate(stages):
                                with st.expander(f"Stage {i+1}: {stage['stage'].replace('_', ' ').title()}", expanded=(i==0)):
                                    stage_output = stage.get('output', {})
                                    
                                    if stage['stage'] == 'ner_coref':
                                        st.markdown("**🏷️ Named Entities & Coreference Clusters:**")
                                        clusters = stage_output.get('clusters', {})
                                        for cid, cluster_data in clusters.items():
                                            st.write(f"**Cluster {cid}:** {cluster_data.get('canonical_name', 'Unknown')}")
                                            refs = cluster_data.get('entity_references', [])
                                            if refs:
                                                st.write(f"  - References: {[ref[0] for ref in refs]}")
                                    
                                    elif stage['stage'] == 'clause_splitting':
                                        st.markdown("**✂️ Text Clauses:**")
                                        clauses = stage_output.get('clauses', [])
                                        for j, clause in enumerate(clauses):
                                            st.write(f"**Clause {j+1}:** `{clause}`")
                                    
                                    elif stage['stage'] == 'node_creation':
                                        st.markdown("**🔗 Graph Nodes Created:**")
                                        details = stage_output.get('details', [])
                                        if details:
                                            node_df = pd.DataFrame(details)
                                            st.dataframe(node_df)
                                        else:
                                            st.warning("No nodes were created")
                                    
                                    elif stage['stage'] == 'edge_creation':
                                        st.markdown("**🔀 Graph Edges Created:**")
                                        details = stage_output.get('details', [])
                                        if details:
                                            edge_df = pd.DataFrame(details)
                                            st.dataframe(edge_df)
                                        else:
                                            st.warning("No edges were created")
                                    
                                    elif stage['stage'] == 'sentiment_calculation':
                                        st.markdown("**😊 Final Sentiment Scores:**")
                                        sentiments = stage_output.get('final_sentiments', {})
                                        if sentiments:
                                            sent_df = pd.DataFrame(list(sentiments.items()), columns=['Entity', 'Score'])
                                            st.dataframe(sent_df)
                                        else:
                                            st.warning("No sentiment scores calculated")
                                    
                                    else:
                                        if stage_output:
                                            st.json(stage_output)
                                        else:
                                            st.info("No output data available for this stage")
                        else:
                            st.info("No pipeline execution stages found in trace data. The execution may not have completed or the trace format may be different.")
                    
                    with analysis_tabs[2]:                   
                        st.markdown("#### 🧩 Individual Module Analysis")
                        
                        for module_name, module_data in modules.items():
                            with st.expander(f"{module_name.replace('_', ' ').title()}", expanded=False):
                                status = module_data.get('status', 'unknown')
                                
                                if status == 'error':
                                    st.error(f"❌ Module failed: {module_data.get('error', 'Unknown error')}")
                                elif status == 'fallback':
                                    st.warning(f"⚠️ Module used fallback method: {module_data.get('error', 'Unknown reason')}")
                                else:
                                    st.success("✅ Module executed successfully")
                                
                                for key, value in module_data.items():
                                    if key not in ['status', 'error']:
                                        st.write(f"**{key}:** {value}")
                    
                    with analysis_tabs[3]:                       
                        st.markdown("#### 📊 Graph Structure Visualization")
                        
                        graph_html_path = exec_trace.get('graph_html_path') or trace_data.get('graph_html_path')
                        if graph_html_path and os.path.exists(graph_html_path):
                            with open(graph_html_path, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=600, scrolling=True)
                        else:
                            st.markdown("**Graph Nodes:**")
                            nodes = exec_trace.get('graph_nodes', []) or trace_data.get('graph_nodes', [])
                            if nodes:
                                node_data = []
                                for node in nodes:
                                    if isinstance(node, (list, tuple)) and len(node) >= 2:
                                        node_id, node_attrs = node[0], node[1]
                                        node_row = {"ID": node_id}
                                        node_row.update(node_attrs)
                                        node_data.append(node_row)
                                if node_data:
                                    st.dataframe(pd.DataFrame(node_data))
                                else:
                                    st.info("Graph nodes format not recognized")
                            else:
                                st.info("No graph nodes available")
                            
                            st.markdown("**Graph Edges:**")
                            edges = exec_trace.get('graph_edges', []) or trace_data.get('graph_edges', [])
                            if edges:
                                edge_data = []
                                for edge in edges:
                                    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                                        from_node, to_node, edge_attrs = edge[0], edge[1], edge[2]
                                        edge_row = {"From": from_node, "To": to_node}
                                        edge_row.update(edge_attrs)
                                        edge_data.append(edge_row)
                                if edge_data:
                                    st.dataframe(pd.DataFrame(edge_data))
                                else:
                                    st.info("Graph edges format not recognized")
                            else:
                                st.info("No graph edges available")
                    
                    with analysis_tabs[4]:                     
                        st.markdown("#### ⚙️ Interactive Parameter Testing")
                        st.markdown("Test different threshold values to see how they affect the prediction:")
                        
                        test_col1, test_col2 = st.columns(2)
                        with test_col1:
                            test_pos_thresh = st.slider("Positive Threshold", 0.0, 1.0, 0.1, 0.01, key="test_pos")
                        with test_col2:
                            test_neg_thresh = st.slider("Negative Threshold", -1.0, 0.0, -0.1, 0.01, key="test_neg")
                        
                        original_score = selected_row['score']
                        new_prediction = "positive" if original_score > test_pos_thresh else "negative" if original_score < test_neg_thresh else "neutral"
                        
                        st.markdown(f"**Original Score:** {original_score:.3f}")
                        st.markdown(f"**New Prediction:** {new_prediction}")
                        st.markdown(f"**Would this fix the error?** {'✅ Yes' if new_prediction == selected_row['gold'] else '❌ No'}")
                    
                    with analysis_tabs[5]:                
                        st.markdown("#### 📥 Export Detailed Trace")
                        st.markdown("Download the complete execution trace for research or debugging:")
                        
                        export_data = {
                            "metadata": {
                                "item_id": trace_data['item_id'],
                                "export_timestamp": pd.Timestamp.now().isoformat(),
                                "pipeline_mode": trace_data['pipeline_mode']
                            },
                            "error_summary": {
                                "aspect_term": selected_row['term'],
                                "gold_label": selected_row['gold'],
                                "predicted_label": selected_row['pred'],
                                "confidence_score": selected_row['score']
                            },
                            "full_trace": trace_data
                        }
                        
                        export_json = json.dumps(export_data, indent=2, default=str)
                        
                        st.download_button(
                            label="📄 Download JSON Trace",
                            data=export_json,
                            file_name=f"error_trace_{trace_data['item_id']}.json",
                            mime="application/json"
                        )
                        
                        summary_text = f"""
ETSA Pipeline Error Analysis Report
==================================

Item ID: {trace_data['item_id']}
Pipeline Mode: {trace_data['pipeline_mode']}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ERROR DETAILS:
- Aspect Term: {selected_row['term']}
- Expected Label: {selected_row['gold']}
- Predicted Label: {selected_row['pred']}
- Confidence Score: {selected_row['score']:.3f}

INPUT TEXT:
{trace_data['input_text']}

PIPELINE EXECUTION SUMMARY:
"""
                        for module_name, module_data in modules.items():
                            summary_text += f"\n{module_name.upper()}:"
                            summary_text += f"\n  Status: {module_data.get('status', 'unknown')}"
                            if 'error' in module_data:
                                summary_text += f"\n  Error: {module_data['error']}"
                            summary_text += "\n"
                        
                        st.download_button(
                            label="📋 Download Summary Report",
                            data=summary_text,
                            file_name=f"error_summary_{trace_data['item_id']}.txt",
                            mime="text/plain"
                        )
                
                else:
                    st.error(f"❌ Trace file not found: {trace_path}")
        else:
            st.success("🎉 No errors found in this benchmark run! All predictions were correct.")

    with tab4:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                st.code(f.read(), language='log')
        else:
            st.warning("Log file (benchmark.log) not found.")
    
    with tab5:
        display_comparison_analysis()



with st.sidebar:
    st.header("⚙️ Controls")
    
    st.subheader("New Benchmark Run")
    dataset_options = [
        "test_laptop_2014", 
        "test_restaurant_2014", 
        "test_laptop_2016", 
        "test_restaurant_2016",
        "unified_2014",                                        
        "unified_2016"                           
    ]
    selected_dataset = st.selectbox("Select Dataset", options=dataset_options, key="sb_dataset")
    
                       
    if selected_dataset == "unified_2014":
        st.info("📊 Unified 2014: Restaurant (54) + Laptop (40) = 94 samples")
    elif selected_dataset == "unified_2016":
        st.info("📊 Unified 2016: Combined restaurant and laptop datasets")
    elif "2014" in selected_dataset:
        if "restaurant" in selected_dataset:
            st.info("📊 Restaurant 2014: 54 samples")
        elif "laptop" in selected_dataset:
            st.info("📊 Laptop 2014: 40 samples")
    elif "2016" in selected_dataset:
        st.info("📊 2016 dataset")
    
    mode_options = [
        "full_stack", 
        "efficiency",
        "no_formulas",
        "vader_baseline", 
        "transformer_absa",
        "ner_basic", 
        "no_modifiers", 
        "no_relations",
        "run_all_modes"
    ]
    selected_mode = st.selectbox("Select Run Mode", options=mode_options, key="sb_mode")
    
    st.caption("full_stack (complete pipeline), efficiency (fast rule-based), no_formulas (ablation test - null averages), vader_baseline (VADER sentiment on all entities), transformer_absa (DeBERTa-v3 end-to-end ABSA), ner_basic (basic NER no coreference), no_modifiers (no modifier extraction), no_relations (no relation extraction), run_all_modes (batch all 8 modes)")
    
    limit = st.number_input("Limit items (0 for all)", min_value=0, value=0, key="sb_limit")

    st.markdown("---")
    st.subheader("Sentiment Thresholds")
    pos_thresh = st.slider("Positive Threshold", 0.0, 1.0, 0.1, 0.01)
    neg_thresh = st.slider("Negative Threshold", -1.0, 0.0, -0.1, 0.01)
    st.markdown("---")

    if st.button("🚀 Run New Benchmark", use_container_width=True):
        execute_benchmark_run(selected_dataset, selected_mode, limit, pos_thresh, neg_thresh)

    st.markdown("---")
    st.header("📂 Past Runs")

    run_dirs = get_run_dirs()
    index = 0
    if st.session_state.selected_run and st.session_state.selected_run in run_dirs:
        index = run_dirs.index(st.session_state.selected_run)

    def on_change_run():
        st.session_state.selected_run = st.session_state.sb_selected_run

    st.selectbox(
        "Select a benchmark to view:", 
        run_dirs, 
        index=index,
        key="sb_selected_run",
        on_change=on_change_run
    )

                                    
    st.markdown("---")
    st.subheader("Preset Group Comparison")
                                                             
    group_options = [
        "All full_stack runs",
        "All vader_baseline runs",
        "All transformer_absa runs",
        "All ner_basic runs",
        "All no_modifiers runs",
        "All no_relations runs",
        "All runs for test_laptop_2014",
        "All runs for test_restaurant_2014",
        "All runs for test_laptop_2016",
        "All runs for test_restaurant_2016",
        "All runs for unified_2014",
        "All runs for unified_2016",
        "None (off)"
    ]
    selected_group = st.selectbox("Select a group to compare:", group_options, index=len(group_options)-1, key="sb_group_compare")
    
                                                                
    if selected_group != "None (off)":
                            
        if selected_group.startswith("All runs for "):
            dataset = selected_group.replace("All runs for ", "")
            group_run_names = [d for d in run_dirs if dataset in d]
        else:
            mode = selected_group.replace("All ", "").replace(" runs", "")
            group_run_names = [d for d in run_dirs if d.endswith(f"_{mode}") or f"_{mode}_" in d]
        if group_run_names:
            st.session_state.batch_runs = group_run_names
            st.session_state.show_batch_comparison = True
            st.info(f"Showing comparison for {len(group_run_names)} runs: {selected_group}")
        else:
            st.warning("No runs found for this group.")
    
                                                                        
    if st.session_state.get('show_batch_comparison', False):
        if st.button("🔄 Clear Group Comparison", use_container_width=True):
            st.session_state.show_batch_comparison = False
            st.session_state.batch_runs = []
            st.rerun()

if st.session_state.get('show_batch_comparison', False):
    display_batch_comparison()
else:
    display_results(st.session_state.selected_run)
